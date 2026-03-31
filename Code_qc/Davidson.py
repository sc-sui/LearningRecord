import numpy as np

def davidson(T, A_diag, x0=None, tol=1e-6, maxiter=200, space=50):
    """
    计算最小近似本征值和对应的本征向量（往往是我们需要的） 
    e:最小近似本征值；r:残差；x0:初猜向量；A：大矩阵；
    T：lambda x: A.dot(x)；h：小矩阵；vs:基；c：单位特征向量
    """
    
    def precond(r, e):
        return r / (A_diag - e + 1e-6)  

    if x0 is None:
        x0 = 1.0 / (A_diag - A_diag.argmin() + 1e-6) # 矩阵的特征值往往靠近对角线的最小元素
        x0 = x0 / np.linalg.norm(x0)

    vs = []
    hv = []
    h = np.empty((space, space))

    for cycle in range(maxiter):
        vs.append(x0) # 存了x个(n,)的一维数组，len=x
        hv.append(T(x0))
        n = len(vs) 

        for i in range(n):
            h[i, n-1] = np.dot(vs[i], hv[n-1])
            h[n-1, i] = h[i, n-1].conj()

        e, c = np.linalg.eigh(h[:n, :n]) 
        e0 = e[0]
        c0 = c[:, 0]
        x = np.zeros_like(x0)
        
        for i, c0[i] in enumerate(c0):
            x += vs[i] * c0[i]
        residual = -e0 * x
        
        for i, c0[i] in enumerate(c0):
            residual += hv[i] * c0[i]

        norm = np.linalg.norm(residual)
        print(f'davidson cycle={cycle} e0={e0:.8f} residual={norm:.5g}')
        
        if norm < tol:
            break
        if len(vs) >= space:
            x0, vs, hv = x, [], [] # 用目前最优解x重启一遍
            continue

        x0 = precond(residual, e0)

        for v in vs:
            x0 -= v * np.dot(v, x0) # 正交化
        norm = np.linalg.norm(x0)
        assert norm > 1e-10, '基向量线性相关'  
        x0 /= norm
    else:
        raise RuntimeError('未收敛')

    return e0, x

if __name__ == "__main__":
    np.random.seed(1) 
    n = 1000
    a = np.random.rand(n, n)
    a = a.T + a
    aop = lambda x: a.dot(x) 
    e = davidson(aop, a.diagonal())[0]
    ref, _ = np.linalg.eigh(a)
    assert abs(e - ref[0]).max() < 1e-8
    print(f"最小近似本征值：{e:.8f}")