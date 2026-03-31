import matplotlib.pyplot as plt
import numpy as np

"""
INPUT U:  (GRAD rho) * GRAD(ABS(GRAD rho))/(rho ** 2 * (2 * KF) ** 3);
INPUT V: (LAPLACIAN rho) / (rho * (2 * KF) ** 2)
rtrs = np.sqrt(rs)
EU = unpolarized LSD correlation energy; EURS= dEU / drs
EP = fully polarized LSD correlation energy; EPRS = dEP / drs
agrad = |∇ρ|; delgrad = ∇·∇ρ; lap = ∇²ρ
Wigner-Seitz radius: n=3 / (4 * pi * rs ^ 3) = (kF ^ 3) / (3 * pi ^ 2)
pi32 = 3 * pi ^ 2;
alpha = (9 * pi / 4) ^ (1 / 3)
Uniform Gas Correlation : LDA_C_PW
"""

def exchpbe(rho, S, U, V, lgga, lpot):
    thrd = 1.0 / 3.0
    thrd4 = 4.0 / 3.0
    pi = 3.14159265358979323846264338327950
    ax = -0.738558766382022405884230032680836
    um = 0.2195149727645171
    uk = 0.8040
    ul = um / uk

    if rho < 1e-18:
        return (0.0, 0.0)

    exunif = ax * (rho ** thrd)

    if lgga == 0:
        ex = exunif
        vx = ex * thrd4
        return (ex, vx)

    S2 = S * S
    P0 = 1.0 + ul * S2
    FxPBE = 1.0 + uk - uk / P0
    ex = exunif * FxPBE

    if lpot == 0:
        return (ex, 0.0)
    if abs(S) > 1e-18:
        Fs = 2.0 * uk * ul / (P0 * P0)
        Fss = -4.0 * ul * S * Fs / P0
    else:
        Fs = 0.0
        Fss = 0.0

    term = (thrd4 * FxPBE) - (U - thrd4 * S2 * S) * Fss - V * Fs
    vx = exunif * term
    
    return (ex, vx)

def GCOR2(A, A1, B1, B2, B3, B4, rtrs):
    Q0 = -2.0 * A * (1.0 + A1 * rtrs * rtrs)
    Q1 = 2.0 * A * rtrs * (B1 + rtrs * (B2 + rtrs * (B3 + B4 * rtrs)))
    
    if abs(Q1) > 1e-18:
        Q2 = np.log(1.0 + 1.0 / Q1)
    else:
        Q2 = 0.0

    GG = Q0 * Q2

    if abs(rtrs) > 1e-18:
        Q3 = A * (B1 / rtrs + 2.0 * B2 + rtrs * (3.0 * B3 + 4.0 * B4 * rtrs))
    else:
        Q3 = 0.0
    if abs(Q1) > 1e-18 and abs(Q1 * (1.0 + Q1)) > 1e-18:
        GGRS = -2.0 * A * A1 * Q2 - Q0 * Q3 / (Q1 * (1.0 + Q1))
    else:
        GGRS = 0.0

    return GG, GGRS

def corpbe(rs, zet, t, uu, vv, ww, lgga, lpot):
    thrd = 1.0 / 3.0
    thrdm = -thrd
    thrd2 = 2.0 * thrd
    sixthm = thrdm / 2.0
    thrd4 = 4.0 / 3.0
    GAM = 0.5198420997897463295344212145565
    fzz = 8.0 / (9.0 * GAM)
    gamma = 0.03109069086965489503494086371273
    bet = 0.06672455060314922
    delt = bet / gamma
    eta = 1e-12

    if rs < 1e-18:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    rtrs = np.sqrt(rs)

    EU, EURS = GCOR2(0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, rtrs)
    EP, EPRS = GCOR2(0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, rtrs)
    ALFM, ALFRSM = GCOR2(0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, rtrs)
    ALFC = -ALFM

    Z4 = zet ** 4
    F = (((1.0 + zet) ** thrd4 + (1.0 - zet) ** thrd4) - 2.0) / GAM
    ec = EU * (1.0 - F * Z4) + EP * F * Z4 - ALFM * F * (1.0 - Z4) / fzz

    ECRS = EURS * (1.0 - F * Z4) + EPRS * F * Z4 - ALFRSM * F * (1.0 - Z4) / fzz

    if abs(1.0 + zet) > 1e-18 and abs(1.0 - zet) > 1e-18:
        FZ = thrd4 * ((1.0 + zet) ** thrd - (1.0 - zet) ** thrd) / GAM
    else:
        FZ = 0.0

    ECZET = 4.0 * (zet ** 3) * F * (EP - EU + ALFM / fzz) + FZ * (Z4 * EP - Z4 * EU - (1.0 - Z4) * ALFM / fzz)
    COMM = ec - rs * ECRS / 3.0 - zet * ECZET
    vcup = COMM + ECZET
    vcdn = COMM - ECZET

    if lgga == 0:
        return (ec, vcup, vcdn, 0.0, 0.0, 0.0)

    G = ((1.0 + zet) ** thrd2 + (1.0 - zet) ** thrd2) / 2.0
    G3 = G ** 3

    if abs(G3 * gamma) > 1e-18:
        PON = -ec / (G3 * gamma)
        denominator_B = np.exp(PON) - 1.0
        if abs(denominator_B) > 1e-18:
            B = delt / denominator_B
        else:
            B = 0.0
    else:
        B = 0.0

    B2 = B * B
    T2 = t * t
    T4 = T2 * T2
    Q4 = 1.0 + B * T2
    Q5 = 1.0 + B * T2 + B2 * T4

    if abs(Q5) > 1e-18:
        h = G3 * (bet / delt) * np.log(1.0 + delt * Q4 * T2 / Q5)
    else:
        h = 0.0

    if lpot == 0:
        return (ec, vcup, vcdn, h, 0.0, 0.0)

    G4 = G3 * G
    T6 = T4 * T2
    RSTHRD = rs / 3.0

    if abs(1.0 + zet) > 1e-18 and abs(1.0 - zet) > 1e-18:
        term1 = ((1.0 + zet) ** 2 + eta) ** sixthm
        term2 = ((1.0 - zet) ** 2 + eta) ** sixthm
        GZ = (term1 - term2) / 3.0
    else:
        GZ = 0.0
    if abs(B) > 1e-18:
        FAC = delt / B + 1.0
    else:
        FAC = 1.0
    if abs(bet * G4) > 1e-18:
        BG = -3.0 * B2 * ec * FAC / (bet * G4)
    else:
        BG = 0.0
    if abs(bet * G3) > 1e-18:
        BEC = B2 * FAC / (bet * G3)
    else:
        BEC = 0.0

    Q8 = Q5 ** 2 + delt * Q4 * Q5 * T2 if abs(Q5) > 1e-18 else 0.0
    Q9 = 1.0 + 2.0 * B * T2

    if abs(Q8) > 1e-18:
        hB = -bet * G3 * B * T6 * (2.0 + B * T2) / Q8
    else:
        hB = 0.0

    hRS = -RSTHRD * hB * BEC * ECRS

    if abs(Q8) > 1e-18:
        FACT0 = 2.0 * delt - 6.0 * B
        FACT1 = Q5 * Q9 + Q4 * Q9 * Q9
        hBT = 2.0 * bet * G3 * T4 * ((Q4 * Q5 * FACT0 - delt * FACT1) / Q8) / Q8
    else:
        hBT = 0.0

    hRST = RSTHRD * T2 * hBT * BEC * ECRS

    if abs(G) > 1e-18:
        hZ = 3.0 * GZ * h / G + hB * (BG * GZ + BEC * ECZET)
    else:
        hZ = 0.0
    if abs(Q8) > 1e-18:
        hT = 2.0 * bet * G3 * Q9 / Q8
    else:
        hT = 0.0
    if abs(G) > 1e-18:
        hZT = 3.0 * GZ * hT / G + hBT * (BG * GZ + BEC * ECZET)
    else:
        hZT = 0.0
    if abs(Q8) > 1e-18:
        FACT2 = Q4 * Q5 + B * T2 * (Q4 * Q9 + Q5)
        FACT3 = 2.0 * B * Q5 * Q9 + delt * FACT2
        hTT = 4.0 * bet * G3 * t * (2.0 * B / Q8 - (Q9 * FACT3 / Q8) / Q8)
    else:
        hTT = 0.0

    COMM = h + hRS + hRST + T2 * hT / 6.0 + 7.0 * T2 * t * hTT / 6.0

    if abs(G) > 1e-18:
        PREF = hZ - GZ * T2 * hT / G
        FACT5 = GZ * (2.0 * hT + t * hTT) / G
    else:
        PREF = 0.0
        FACT5 = 0.0

    COMM = COMM - PREF * zet - uu * hTT - vv * hT - ww * (hZT - FACT5)

    dvcup = COMM + PREF
    dvcdn = COMM - PREF
    return (ec, vcup, vcdn, h, dvcup, dvcdn)

def easypbe(up, agrup, delgrup, uplap, dn, agrdn, delgrdn, dnlap, agr, delgr, lcor, lpot):
    thrd = 1.0 / 3.0
    thrd2 = 2.0 * thrd
    pi32 = 29.608813203268075856503472999628
    pi = 3.1415926535897932384626433832795
    alpha = 1.91915829267751300662482032624669

    exuplsd = 0.0
    vxuplsd = 0.0
    exuppbe = 0.0
    vxuppbe = 0.0

    exdnlsd = 0.0
    vxdnlsd = 0.0
    exdnpbe = 0.0
    vxdnpbe = 0.0

    rho2 = 2.0 * up
    if rho2 > 1e-18:
        fk = (pi32 * rho2) ** thrd
        s = agrup / (fk * rho2) if abs(fk * rho2) > 1e-18 else 0.0
        u = 4.0 * delgrup / (rho2 ** 2 * (2.0 * fk) ** 3) if abs(rho2 ** 2 * (2.0 * fk) ** 3) > 1e-18 else 0.0
        v = 2.0 * uplap / (rho2 * (2.0 * fk) ** 2) if abs(rho2 * (2.0 * fk) ** 2) > 1e-18 else 0.0

        exuplsd, vxuplsd = exchpbe(rho2, s, u, v, 0, lpot)
        exuppbe, vxuppbe = exchpbe(rho2, s, u, v, 1, lpot)

    rho2 = 2.0 * dn
    if rho2 > 1e-18:
        fk = (pi32 * rho2) ** thrd
        s = agrdn / (fk * rho2) if abs(fk * rho2) > 1e-18 else 0.0
        u = 4.0 * delgrdn / (rho2 ** 2 * (2.0 * fk) ** 3) if abs(rho2 ** 2 * (2.0 * fk) ** 3) > 1e-18 else 0.0
        v = 2.0 * dnlap / (rho2 * (2.0 * fk) ** 2) if abs(rho2 * (2.0 * fk) ** 2) > 1e-18 else 0.0

        exdnlsd, vxdnlsd = exchpbe(rho2, s, u, v, 0, lpot)
        exdnpbe, vxdnpbe = exchpbe(rho2, s, u, v, 1, lpot)

    rho = up + dn
    
    if rho > 1e-18:
        exlsd = (exuplsd * up + exdnlsd * dn) / rho
        expbe = (exuppbe * up + exdnpbe * dn) / rho
    else:
        exlsd = 0.0
        expbe = 0.0
    if rho < 1e-18:
        return (exlsd, vxuplsd, vxdnlsd, 0.0, 0.0, 0.0, expbe, vxuppbe, vxdnpbe, 0.0, 0.0, 0.0)

    zet = (up - dn) / rho
    g = ((1.0 + zet) ** thrd2 + (1.0 - zet) ** thrd2) / 2.0
    fk = (pi32 * rho) ** thrd
    rs = alpha / fk
    sk = np.sqrt(4.0 * fk / pi)
    twoksg = 2.0 * sk * g
    t = agr / (twoksg * rho) if abs(twoksg * rho) > 1e-18 else 0.0
    uu = delgr / (rho ** 2 * twoksg ** 3) if abs(rho ** 2 * twoksg ** 3) > 1e-18 else 0.0
    rholap = uplap + dnlap
    vv = rholap / (rho * twoksg ** 2) if abs(rho * twoksg ** 2) > 1e-18 else 0.0

    denominator_ww = (rho * twoksg) ** 2 if abs(rho * twoksg) > 1e-18 else 1e-30
    ww = (agrup ** 2 - agrdn ** 2 - zet * agr ** 2) / denominator_ww

    ec, vcup, vcdn, h, dvcup, dvcdn = corpbe(rs, zet, t, uu, vv, ww, 1, lpot)

    eclsd = ec
    ecpbe = ec + h
    vcuplsd = vcup
    vcdnlsd = vcdn
    vcuppbe = vcup + dvcup
    vcdnpbe = vcdn + dvcdn

    return (exlsd, vxuplsd, vxdnlsd, eclsd, vcuplsd, vcdnlsd, expbe, vxuppbe, vxdnpbe, ecpbe, vcuppbe, vcdnpbe)

def main():
    DDN = 0
    agrdn = 0.0

    dup = np.linspace(0.0, 1.0, 1000)
    agrup = 0.3

    sum_ex_lsd = []
    sum_ex_pbe = []
    sum_ec_lsd = []
    sum_ec_pbe = []

    for DUP in dup:
        delgrup = 0
        uplap = 0
        delgrdn = 0
        dnlap = 0

        D = DUP + DDN
        agrad = agrup + agrdn
        delgrad = delgrup + delgrdn

        exlsd, _, _, eclsd, _, _, expbe, _, _, ecpbe, _, _ = easypbe(
            DUP, agrup, delgrup, uplap, DDN, agrdn, delgrdn, dnlap, agrad, delgrad, 1, 1
        )

        sum_ex_lsd.append(D * exlsd)
        sum_ex_pbe.append(D * expbe)
        sum_ec_lsd.append(D * eclsd)
        sum_ec_pbe.append(D * ecpbe)

    sum_ex_lsd = np.array(sum_ex_lsd)
    sum_ex_pbe = np.array(sum_ex_pbe)
    sum_ec_lsd = np.array(sum_ec_lsd)
    sum_ec_pbe = np.array(sum_ec_pbe)

    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (16, 12),
        'axes.linewidth': 1.2
    })

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,12))

    ax1.plot(dup, sum_ex_lsd, color='#2E86AB', linewidth=2.5)
    ax1.set_title('LSD Exchange Energy (exlsd)')
    ax1.set_xlabel('Spin-up density')
    ax1.set_ylabel('Energy (a.u.)')
    ax1.grid(True)

    ax2.plot(dup, sum_ex_pbe, color='#E63946', linewidth=2.5)
    ax2.set_title('PBE Exchange Energy (expbe)')
    ax2.set_xlabel('Spin-up density')
    ax2.set_ylabel('Energy (a.u.)')
    ax2.grid(True)

    ax3.plot(dup, sum_ec_lsd, color='#F77F00', linewidth=2.5)
    ax3.set_title('LSD Correlation Energy (eclsd)')
    ax3.set_xlabel('Spin-up density')
    ax3.set_ylabel('Energy (a.u.)')
    ax3.grid(True)

    ax4.plot(dup, sum_ec_pbe, color='#7209B7', linewidth=2.5)
    ax4.set_title('PBE Correlation Energy (ecpbe)')
    ax4.set_xlabel('Spin-up density')
    ax4.set_ylabel('Energy (a.u.)')
    ax4.grid(True)

    plt.tight_layout()

    plt.savefig('PBE-LSDA.png', dpi=300, bbox_inches='tight')
    plt.close()  

if __name__ == "__main__":
    main()