from __future__ import print_function, division
import sys
import numpy as np
import ql
import babusca.analyze as analyze
import babusca.g2 as g2
import babusca.tmatrix as tmatrix
import babusca.smatrix as smatrix
import matplotlib.pyplot as plt


# test g1
def s1_num(d, phi, chli=0, chlo=1, deltas=0, offset=1e3):

    omegas = [offset - d, offset + d]
    Us = [0, 0]
    se = ql.QL(2, omegas, Us, xs=[0, phi / (offset)], gs=[1 / np.sqrt(np.pi), 1 / np.sqrt(np.pi)])

    _, S = smatrix.one_particle(se, chli, chlo, deltas + offset)

    return S


def g2_exact_00(d, phi, delta, taus):
    # For U infinite
    # Gamma = 1
    Gam = 1

    # detunings
    dls = [delta + d, delta - d]

    # eigenvalues
    lam = [0] * 2
    lam[0] = dls[0] + 2 * 1j * Gam
    lam[1] = dls[1] + 2 * 1j * Gam

    m1 = (lam[0] + lam[1]) / 2
    m2 = np.sqrt(((lam[0] - lam[1]) / 2) ** 2 - 4 * Gam ** 2 * np.exp(2 * 1j * phi))

    als = [m1 + m2, m1 - m2]

    cs = [0] * 2
    cs[0] = (2 * als[0] - (dls[0] + dls[1] + 2 * 1j * Gam * (1 - np.exp(2 * 1j * phi)))) / (als[0] - als[1])
    cs[1] = (2 * als[1] - (dls[0] + dls[1] + 2 * 1j * Gam * (1 - np.exp(2 * 1j * phi)))) / (-als[0] + als[1])

    m0 = cs[0] / als[0] + cs[1] / als[1]

    intr = - 4 * Gam ** 2 * m0 * (
        (1 / sum(lam) - 1 / als[0]) * cs[0] * np.exp(1j * als[0] * taus)
        +
        (1 / sum(lam) - 1 / als[1]) * cs[1] * np.exp(1j * als[1] * taus)
    )

    # g1 = g1_exact(d, phi, delta, gamma=Gam)
    s1 = s1_num(d, phi, 0, 0, np.array([delta]))

    return np.abs(1 + 1 / (s1 ** 2) * intr) ** 2
    # return np.abs(intr)


def g2s_exact_00(d, phi, deltas, taus):
    g2s = np.zeros((len(deltas), len(taus)), dtype=np.float64)
    for i, delta in enumerate(deltas):
        g2s[i, :] = g2_exact_00(d, phi, delta, taus)

    return g2s


def g2s_num_00(d, phi, U, offsets, taus):
    offset = 1e3   # offset
    omegas = [0] * 2
    Us = [U, U]
    se = ql.QL(2, omegas, Us, xs=[0, phi / (offset)])

    g2s = np.zeros((len(offsets), len(taus)), dtype=np.float64)
    for i, off in enumerate(offsets):
        se.reset()
        se.model.omegas = [offset - off - d, offset - off + d]
        res = g2.coherent_state(se, (0, 0), (0, 0), 2 * offset, taus)
        g2s[i, :] = res['g2']

    return g2s


def g2_test(d, phi):
    # deltas = np.linspace(-10, 10, 255)
    deltas = np.array([0])
    taus = np.linspace(0, 10, 255)

    g2n = g2s_num_00(d, phi, 1e8, deltas, taus)
    g2s = g2s_exact_00(d, phi, deltas, taus)

    plt.semilogy(taus, g2n.T, label='num')
    plt.semilogy(taus, g2s.T, label='exc', ls=':')

    plt.legend()

    plt.tight_layout()
    plt.show()


def check_eigenvalues(d, phi, off):

    Gam = 1

    # detunings
    dls = [off + d, off - d]

    # eigenvalues
    lam = [0] * 2
    lam[0] = dls[0] + 2 * 1j * Gam
    lam[1] = dls[1] + 2 * 1j * Gam

    m1 = (lam[0] + lam[1]) / 2
    m2 = np.sqrt(((lam[0] - lam[1]) / 2) ** 2 - 4 * Gam ** 2 * np.exp(2 * 1j * phi))

    als = [m1 + m2, m1 - m2]
    print(als)

    offset = 1e3   # offset
    omegas = [offset - off - d, offset - off + d]
    Us = [1e8, 1e8]
    se = ql.QL(2, omegas, Us, xs=[0, phi / (offset)])

    Es = se.eigenenergies(1, offset)
    print([-(E - offset) for E in Es])


if __name__ == '__main__':
    g2_test(2.2, .2)
    # check_eigenvalues(2.03, 0.8, 0.3)
    # g1_test(1, 0)
    # g2s_num_00(1, 0, 0, 0)
    # asd(1, 1e8, 0)
