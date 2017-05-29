from __future__ import division, print_function
import sys
sys.path.append('../')
import numpy as np
import scattering
import tmatrix
import ttmatrix
import matplotlib.pyplot as plt


def test_tmatrix_oneparticle():
    """
    Test the S(1) matrix for one example system
    """
    m = scattering.Model([0]*2, [[0, 1, 1]], [0]*2)
    channels = [
        scattering.Channel(site=0, strength=1),
        scattering.Channel(site=1, strength=1),
    ]
    s = scattering.Setup(m, channels)
    E = 0
    _, T1 = tmatrix.one_particle(s, 0, 1, np.array([E]))

    T10num = 2*np.pi*np.abs(T1[0])
    T10ana = 2*np.pi/(1+(np.pi)**2)

    assert np.isclose(T10num, T10ana, 1e-3), \
        'T(1) fails for two-sites coupled in series. T(1) = {0} != {1}.'.format(T10num, T10ana)


def test_tmatrix_oneparticle2():
    m = scattering.Model([0]*2, [[0, 1, 1]], [0]*2)
    channels = [
        scattering.Channel(site=0, strength=1),
        scattering.Channel(site=0, strength=1),
    ]
    s = scattering.Setup(m, channels)
    E = 0
    _, T1 = tmatrix.one_particle(s, 0, 0, np.array([E]))
    assert(np.isclose(np.abs(T1[0])**2, 0, 1e-3))

    _, T1 = tmatrix.one_particle(s, 0, 1, np.array([E]))
    assert np.isclose(np.abs(T1[0])**2, 0, 1e-3), \
        'T(1) fails for two side coupled sites. T(1) = {0} != 0.'.format(np.abs(T1[0])**2)


def test_tmatrix_two_particle_non_interacting():
    m = scattering.Model([0]*2, [[0, 1, 1]], [0]*2)
    channels = [
        scattering.Channel(site=0, strength=.1),
        scattering.Channel(site=1, strength=.1),
    ]
    se = scattering.Setup(m, channels)

    E = 0
    dE = 0
    chls = [[0, 0], [0, 1], [1, 0], [1, 1]]
    qs = np.linspace(-5, 5, 512)

    # T2as = np.zeros((4, 4, len(qs)), np.complex128)
    # T2bs = np.zeros((4, 4, len(qs)), np.complex128)
    for ci in xrange(4):
        for co in xrange(4):
            _, T2a, T2b = tmatrix.two_particle(se,
                                               chlsi=chls[ci],
                                               chlso=chls[co],
                                               E=E,
                                               dE=dE,
                                               qs=qs
                                               )

            plt.subplot(4, 4, 1 + ci + co*4)
            plt.plot(qs, np.real(T2a), label='real'.format(chls[ci], chls[co]))
            plt.plot(qs, np.imag(T2a))
            plt.title('{}-{}'.format(chls[ci], chls[co]))
            plt.legend()
    plt.show()

    # Test if S2 vanishes
    # assert np.allclose(np.ones((4, 4)) + S2s, np.ones((4, 4)), 1e-2), \
        # 'S2 does not vanish for non-interacting systems, S2 = {0}.'.format(np.abs(S2s))


def test_ttmatrix():

    m = scattering.Model([0]*2, [[0, 1, 1]], [0]*2)
    channels = [
        scattering.Channel(site=0, strength=.1),
        scattering.Channel(site=1, strength=.1),
    ]
    se = scattering.Setup(m, channels)

    E = 0
    dE = 0
    chls = [[0, 0], [0, 1], [1, 0], [1, 1]]
    qs = np.linspace(-5, 5, 512)

    # T2as = np.zeros((4, 4, len(qs)), np.complex128)
    # T2bs = np.zeros((4, 4, len(qs)), np.complex128)
    for ci in xrange(4):
        for co in xrange(4):
            _, T2a, _, T2b = ttmatrix.two_particle(se,
                                                chlsi=chls[ci],
                                                chlso=chls[co],
                                                E=E,
                                                dE=dE,
                                                qs=qs
                                                )

            plt.subplot(4, 4, 1 + ci + co*4)
            plt.plot(qs, np.real(T2a + T2b), label='real'.format(chls[ci], chls[co]))
            plt.plot(qs, np.imag(T2a + T2b))
            plt.title('{}-{}'.format(chls[ci], chls[co]))
            plt.legend()
    plt.show()


def test_tmatrix_vs_ttmatrix():

    m = scattering.Model([0]*2, [[0, 1, 1]], [0]*2)
    channels = [
        scattering.Channel(site=0, strength=.1),
        scattering.Channel(site=1, strength=.1),
    ]
    se = scattering.Setup(m, channels)

    E = 0
    dE = 0
    chlsi = (1, 1)
    chlso = (1, 1)
    qs = np.array([.3])

    qs, T2, N2 = tmatrix.two_particle(se,
                                      chlsi=chlsi,
                                      chlso=chlso,
                                      E=E, dE=dE,
                                      qs=qs,
                                      verbal=True)

    ks, A2, B2, C2 = ttmatrix.two_particle(se,
                                           chlsi=chlsi,
                                           chlso=chlso,
                                           E=E, dE=dE,
                                           qs=qs,
                                           verbal=True)

    print('Result:')
    print(T2)
    print(A2 + C2)

    print('Interacting:')
    print(T2 - N2)
    print(A2)

    print('Non-interacting:')
    print(N2)
    print(C2)


if __name__ == '__main__':
    # test_tmatrix_oneparticle()
    # test_tmatrix_oneparticle2()
    test_tmatrix_two_particle_non_interacting()
    # test_ttmatrix()
    # test_tmatrix_vs_ttmatrix()