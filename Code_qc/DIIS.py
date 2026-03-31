from collections import deque
from typing import Union
from functools import lru_cache
import json
import numpy as np
import h5py
import scipy.linalg

class DIIS:
    # errvecs:误差向量; tvecs:试验向量
    def __init__(self, filename, max_space=8):
        self.filename = filename
        self.keys = deque(maxlen=max_space)
        self.head = 0

    @property
    def max_space(self):
        return self.keys.maxlen

    def update(self, errvec, tvec):
        with h5py.File(self.filename, mode='a') as f:
            head, self.head = self.head, (self.head + 1) % self.max_space
            self.keys.append(head)

            if f'e{head}' in f:
                f[f'e{head}'][:] = errvec.ravel()
                f[f't{head}'][:] = tvec
            else:
                f[f'e{head}'] = errvec.ravel()
                f[f't{head}'] = tvec
            if 'metadata' in f:
                del f['metadata']
            f['metadata'] = self.dumps()
            f.flush()

            errvecs = [f[f'e{key}'][()] for key in self.keys]
            tvecs = [f[f't{key}'][()] for key in self.keys]
            return extrapolate(errvecs, tvecs)

    def dumps(self):
        return json.dumps({
            'max_space': self.max_space,
            'keys': list(self.keys),
            'head': self.head,
        })

    def save(self):
        with h5py.File(self.filename, mode='a') as f:
            f['metadata'] = self.dumps()


def extrapolate(errvecs, tvecs):
    space = len(tvecs)
    B = np.zeros((space+1, space+1))
    B[-1,:-1] = B[:-1,-1] = 1.
    g = np.zeros(space+1)
    g[-1] = 1
    for i, e1 in enumerate(errvecs):
        for j, e2 in enumerate(errvecs):
            if j < i:
                continue
            B[i,j] = B[j,i] = e1.dot(e2)

    c = scipy.linalg.solve(B, g, assume_a='sym')[:-1]
    sol = tvecs[0] * c[0]
    for v, x in zip(tvecs[1:], c[1:]):
        sol += v * x
    return sol

