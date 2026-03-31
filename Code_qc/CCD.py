import tempfile
from typing import Dict, Union
import numpy as np
import h5py
import pyscf
from pyscf.cc import ccd
from DIIS import DIIS

def einsum(*args):
    return np.einsum(*args, optimize=True)
    
""" 
t2(nvir, nvir, nocc, nocc)-双激发算符
o-占据轨道；v-非占据轨道
"""
def update_CCD_amplitudes(H: Union[Dict, h5py.Group], t2: np.array):
    nvir, nocc = t2.shape[1:3]
    fock = np.asarray(H['fock'])
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]
    e_o = foo.diagonal() #提取HF对角元
    e_v = fvv.diagonal()
    
    H_oovv = np.asarray(H['oovv'])
    Fvv = fvv - .5 * einsum('klcd,bdkl->bc', H_oovv, t2)
    Foo = foo + .5 * einsum('klcd,cdjl->kj', H_oovv, t2)
    Fvv[np.diag_indices(nvir)] -= e_v #减去HF的能量
    Foo[np.diag_indices(nocc)] -= e_o

    H_vvoo = np.asarray(H['vvoo'])
    t2out =  1 * H_vvoo
    H_vvvv = np.asarray(H['vvvv'])
    t2out += 0.5 * einsum('abcd,cdij->abij',H_vvvv,t2)
    H_oooo = np.asarray(H['oooo'])
    t2out += 0.5 * einsum('ijkl,abkl->abij',H_oooo,t2)
    H_vovo = np.asarray(H['vovo'])
    t2out -= (einsum('bkcj,acik->abij',H_vovo,t2) - einsum('bkci,acjk->abij',H_vovo,t2) 
              - einsum('akcj,bcik->abij',H_vovo,t2) + einsum('akci,bcjk->abij',H_vovo,t2))
    t2out += ((-1) * einsum('jk,abik->abij',foo,t2) + einsum('ik,abjk->abij',foo,t2) 
              + einsum('bc,acij->abij',fvv,t2) - einsum('ac,bcij->abij',fvv,t2))
    t2out += (einsum('jk,abik->abij',np.diag(e_o),t2) - einsum('ik,abjk->abij',np.diag(e_o),t2)
              - einsum('bc,acij->abij',np.diag(e_v),t2) + einsum('ac,bcij->abij',np.diag(e_v),t2))
    
    H_oovv = np.asarray(H['oovv'])
    t2out += (0.25 * einsum('klcd,cdij,abkl->abij',H_oovv,t2,t2) - 0.5 * einsum('klcd,acij,bdkl->abij',H_oovv,t2,t2) 
              - 0.5 * einsum('klcd,bdij,ackl->abij',H_oovv,t2,t2) - 0.5 * einsum('klcd,abik,cdjl->abij',H_oovv,t2,t2)
              - 0.5 * einsum('klcd,cdik,abjl->abij',H_oovv,t2,t2) + einsum('klcd,acik,bdjl->abij',H_oovv,t2,t2) 
              + einsum('klcd,bdik,acjl->abij',H_oovv,t2,t2)) 

    t2out /= e_o + e_o[:,None] - e_v[:,None,None] - e_v[:,None,None,None]
    
    return t2out

def get_CCD_corr_energy(H, t2):
    return 0.25 * einsum('ijab,abij->', np.asarray(H['oovv']), t2)

def mo_integrals(mol: pyscf.gto.Mole, orbitals, Hfile=None):
    '''
    <pq||rs>
    no:总电子数；nmo:分子轨道总数; nmo*2:自旋轨道总数
    '''
    no = mol.nelectron 
    nmo = orbitals.shape[1] 
    eri = np.zeros([nmo*2]*4) 
    eri[ ::2, ::2, ::2, ::2] = eri[ ::2, ::2,1::2,1::2] = \
    eri[1::2,1::2, ::2, ::2] = eri[1::2,1::2,1::2,1::2] = \
        pyscf.ao2mo.kernel(mol, orbitals, compact=False).reshape([nmo]*4) #αααα、ααββ、ββαα、ββββ
    eri = eri.transpose(0,2,1,3) - eri.transpose(2,0,1,3)

    if Hfile is None:
        Hfile = tempfile.mktemp()
    with h5py.File(Hfile, 'w') as H:
        H['vvoo'] = vvoo = eri[no:,no:,:no,:no]
        H['oovv'] = vvoo.conj().transpose(2,3,0,1)
        H['vovo'] = eri[no:,:no,no:,:no]
        H['oooo'] = eri[:no,:no,:no,:no]
        H['vvvv'] = eri[no:,no:,no:,no:]

        hcore = pyscf.scf.hf.get_hcore(mol)
        hcore = einsum('pq,pi,qj->ij', hcore, orbitals, orbitals)
        hcore_mo = np.zeros([nmo*2]*2)
        hcore_mo[::2,::2] = hcore_mo[1::2,1::2] = hcore
        H['fock'] = hcore_mo + einsum('ipiq->pq', eri[:no,:,:no,:])
    
    return Hfile

def mp2(H):
    # MP2振幅方程就是CCD振幅方程去掉耦合项，用作初猜
    nocc = H['oooo'].shape[0]
    fock = np.asarray(H['fock'])
    e_o = fock.diagonal()[:nocc]
    e_v = fock.diagonal()[nocc:]
    eijab = e_o + e_o[:,None] - e_v[:,None,None] - e_v[:,None,None,None]
    t2 = np.asarray(H['vvoo']) / eijab
    e = get_CCD_corr_energy(H, t2)
    
    return e, t2

def CCD_solve(mf: pyscf.scf.hf.RHF, conv_tol=1e-7, max_cycle=100):
    mol = mf.mol
    orbitals = mf.mo_coeff
    e_hf = mf.e_tot

    with tempfile.TemporaryDirectory() as tmpdir:
        Hfile = mo_integrals(mol, orbitals, f'{tmpdir}/H')
        diis = DIIS(f'{tmpdir}/diis')
        e_ccd = e_hf
        with h5py.File(Hfile, 'r') as H:
            e_corr, t2 = mp2(H) 
            e_ccd = e_hf + e_corr
            print(f'E(MP2)={e_ccd}')

            for cycle in range(max_cycle):
                t2, t2_prev = update_CCD_amplitudes(H, t2), t2
                e_ccd, e_prev = get_CCD_corr_energy(H, t2) + e_hf, e_ccd
                print(f'{cycle=}, E(CCD)={e_ccd}, dE={e_ccd-e_prev}')
                if abs(t2 - t2_prev).max() < conv_tol:
                    break
                t2 = diis.update(t2 - t2_prev, t2)
    
    return e_ccd

if __name__ == '__main__':
    mol = pyscf.M(atom='N 0. 0 0; N 1.5 0 0', basis='cc-pvdz')
    mf = mol.RHF().run(verbose = 0)
    pyccd = ccd.CCD(mf).run(verbose = 0)
    print(f"PyCCD = {pyccd.e_tot}")
    e_ccd = CCD_solve(mf)
    