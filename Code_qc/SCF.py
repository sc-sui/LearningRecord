from collections import deque
from functools import lru_cache
import tempfile
import pickle
import numpy as np
import scipy.linalg
import h5py
from DIIS import DIIS
from pyscf import gto, scf

def einsum(*args):
    return np.einsum(*args, optimize=True)

class SCFWavefunction:
    gtos = None
    orbitals = None
    energies = None
    occupancies = None
    density_matrices = None

def scf_iter(model, wfn: SCFWavefunction = None):
    hcore, s = model.get_hcore_s()
    if wfn is None:
        wfn = model.get_initial_guess()
    converged = False
    while not converged:
        f = model.get_fock(wfn, hcore, s)
        mo_orbitals, mo_energies = model.eigen(f, s)
        wfn, wfn_old = model.make_wfn(mo_orbitals, mo_energies), wfn
        if model.check_convergence(wfn, wfn_old):
            converged = True
    return wfn

class RHF:
    def __init__(self, mol):
        self.mol = mol
        self.gtos = mol.basis
        self.threshold = 1e-7
        self.chkfile = tempfile.mktemp()
        diisfile = tempfile.mktemp()
        print(f'Checkpoint file is {self.chkfile}. DIIS is saved in {diisfile}')
        self.diis = CDIIS(diisfile)

    @lru_cache
    def get_hcore_s(self):
        t = self.mol.intor('int1e_kin')
        v = self.mol.intor('int1e_nuc')
        s = self.mol.intor('int1e_ovlp')
        return t+v, s

    def get_initial_guess(self):
        h, s = self.get_hcore_s()
        c, e = self.eigen(h, s)
        return self.make_wfn(c, e)

    @property
    @lru_cache
    def eri_tensor(self):
        eri = self.mol.intor('int2e')
        return eri

    @lru_cache(2)
    def get_jk(self, wfn):
        dm = wfn.density_matrices
        j = einsum('ijkl,kl->ij', self.eri_tensor, dm)
        k = einsum('ikjl,kl->ij', self.eri_tensor, dm)
        return j, k

    def get_veff(self, wfn):
        j, k = self.get_jk(wfn)
        return j - k * 0.5

    def get_fock(self, wfn, hcore, s):
        veff = self.get_veff(wfn)
        f = hcore + veff
        self.diis.update(f, wfn.density_matrices, s)
        return f

    def eigen(self, fock, overlap):
        e, c = scipy.linalg.eigh(fock, overlap)
        return c, e

    def make_wfn(self, orbitals, energies):
        wfn = RestrictedCloseShell(self, orbitals, energies)
        with open(self.chkfile, 'wb') as f:
            pickle.dump({'wfn': wfn, 'mol': self.mol, 'gtos': self.gtos}, f)
        return wfn

    def check_convergence(self, wfn, wfn_old):
        t = self.threshold 
        return (np.linalg.norm(wfn.density_matrices - wfn_old.density_matrices) < t
                and np.linalg.norm(self.orbital_gradients(wfn)) < t
                and abs(self.total_energy(wfn) - self.total_energy(wfn_old)) < t)

    def total_energy(self, wfn):
        hcore, s = self.get_hcore_s()
        j, k = self.get_jk(wfn)
        dm = wfn.density_matrices
        e = einsum('ij,ji', hcore, dm)
        e += einsum('ij,ji', j, dm) * 0.5
        e -= einsum('ij,ji', k, dm) * 0.25
        e += self.mol.energy_nuc()
        return e

    def orbital_gradients(self, wfn):
        hcore, s = self.get_hcore_s()
        fock = self.get_fock(wfn, hcore, s)
        fock_mo = wfn.orbitals.T.dot(fock).dot(wfn.orbitals) 
        return fock_mo[(wfn.occupancies!=0) & (wfn.occupancies[:,None]==0)] * 2.0 # 取出Fia(i占据,a非占据)


class RestrictedCloseShell(SCFWavefunction):
    def __init__(self, mf, orbitals, energies):
        self.gtos = mf.gtos
        self.orbitals = orbitals
        self.energies = energies
        nelec = mf.mol.nelectron
        nocc = min(nelec // 2, len(energies)) 
        self.occupancies = np.zeros_like(energies)
        self.occupancies[:nocc] = 2.0

    @property
    @lru_cache
    def density_matrices(self):
        c = self.orbitals
        return (c * self.occupancies).dot(c.T)

class CDIIS(DIIS):
    def __init__(self, filename, max_space=8):
        super().__init__(filename, max_space)
        self.c = None 
    
    # f:FOCK矩阵; d:密度矩阵; s:重叠矩阵
    def update(self, f, d, s):
        with h5py.File(self.filename, mode='a') as h5f:
            if self.c is None:
                _, self.c = scipy.linalg.eigh(s)
                h5f['c'] = self.c
            errvec = f.dot(d).dot(s)
            errvec = errvec - errvec.T 
            # 转为正交基
            errvec = self.c.T.dot(errvec).dot(self.c).ravel()

            head, self.head = self.head, (self.head + 1) % self.max_space
            self.keys.append(head)
            if f'e{head}' in h5f:
                h5f[f'e{head}'][:] = errvec
                h5f[f't{head}'][:] = f
            else:
                h5f[f'e{head}'] = errvec
                h5f[f't{head}'] = f
            if 'metadata' in h5f:
                del h5f['metadata']
            h5f['metadata'] = self.dumps()
            h5f.flush()

            errvecs = [h5f[f'e{key}'][()] for key in self.keys]
            space = len(self.keys)
            B = np.zeros((space+1, space+1))
            B[-1,:-1] = B[:-1,-1] = 1.
            for i, e1 in enumerate(errvecs):
                for j, e2 in enumerate(errvecs):
                    if j < i:
                        continue
                    B[i,j] = B[j,i] = e1.dot(e2)

            #cond(B): λmax / λmin(越大越接近奇异，无法求逆)
            while np.linalg.cond(B) > 1e12:
                B = B[1:,1:]
                self.keys.popleft()

            g = np.zeros(len(self.keys)+1)
            g[-1] = 1
            c = scipy.linalg.solve(B, g, assume_a='sym')[:-1]

            sol = np.zeros_like(f)
            for key, x in zip(self.keys, c):
                sol += h5f[f't{key}'][:] * x
            return sol


if __name__ == "__main__":
    mol = gto.Mole()
    mol.atom = '''
    H 0 0 0
    H 0 0 0.74
    '''
    mol.basis = 'sto-3g'
    mol.build()
   
    rhf = RHF(mol)
    wfn = scf_iter(rhf)
    print(f"Myscf energy: {rhf.total_energy(wfn):.14f} Ha")
    
    mf = scf.RHF(mol)
    mf.run(verbose = 0)
    print(f"Pyscf energy: {mf.e_tot:.14f} Ha")
    
