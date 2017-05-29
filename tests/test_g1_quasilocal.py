from __future__ import print_function, division
import sys
import numpy as np
import ql
# import babusca.analyze as analyze
import babusca.g2 as g2
import babusca.tmatrix as tmatrix
import babusca.smatrix as smatrix
import matplotlib.pyplot as plt


# test g1
def g1_num(d, phi, chli=0, chlo=1, deltas=0, offset=1e3):

    omegas = [offset - d, offset + d]
    Us = [0, 0]
    se = ql.QL(2, omegas, Us, xs=[0, phi / (offset)])

    _, S = smatrix.one_particle(se, chli, chlo, deltas + offset)

    return np.abs(S) ** 2


def g1_exact(d, phi, deltas, gamma=2):

    delta1 = deltas - d
    delta2 = deltas + d

    return np.abs((delta1 / (delta1 + 1j * gamma) * delta2 / (delta2 + 1j * gamma)) /
                  (1 + gamma ** 2 / ((delta1 + 1j * gamma) * (delta2 + 1j * gamma)) * np.exp(2 * 1j * phi))) ** 2


def g1_test(d, phi):

    deltas = np.linspace(-20, 20, 256)
    tnum = g1_num(d, phi, 0, 0, deltas)
    texc = g1_exact(d, phi, deltas)

    assert np.allclose(tnum, texc, rtol=1e-2), "g1: Numerical does not follow exact results"

    plt.plot(deltas, tnum, label='num')
    plt.plot(deltas, texc, label='exc', ls=':')
    plt.legend()
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # test1(2)
    # test1plot(2)
    g1_test(2.3, 1)
    # g2_test(0, .1, .3)
    # g2_test(0, 0)
    # test_centre(0, 1e-5, 1)
