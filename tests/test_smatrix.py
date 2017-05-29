from __future__ import division, print_function
import sys
sys.path.append('../')
import numpy as np
import scattering
import smatrix


def test_smatrix_oneparticle():
    """
    Test the S(1) matrix for one example system
    """
    m = scattering.Model([0] * 2, [[0, 1, 1]], [0] * 2)
    channels = [
        scattering.Channel(site=0, strength=1),
        scattering.Channel(site=1, strength=1),
    ]
    s = scattering.Setup(m, channels)
    E = 0
    _, S1 = smatrix.one_particle(s, 0, 1, np.array([E]))

    S10num = np.abs(S1[0]) ** 2
    S10ana = (2 * np.pi / (1 + (np.pi) ** 2)) ** 2

    assert np.isclose(S10num, S10ana, 1e-3), \
        'S(1) fails for two-sites coupled in series. S(1) = {0} != {1}.'.format(S10num, S01ana)


def test_smatrix_oneparticle2():
    m = scattering.Model([0] * 2, [[0, 1, 1]], [0] * 2)
    channels = [
        scattering.Channel(site=0, strength=1),
        scattering.Channel(site=0, strength=1),
    ]
    s = scattering.Setup(m, channels)
    E = 0
    _, S1 = smatrix.one_particle(s, 0, 0, np.array([E]))
    assert(np.isclose(np.abs(S1[0]) ** 2, 1, 1e-3))

    _, S1 = smatrix.one_particle(s, 0, 1, np.array([E]))
    assert np.isclose(np.abs(S1[0]) ** 2, 0, 1e-3), \
        'S(1) fails for two sites which are sidecoupled. S(1) = {0} != 0.'.format(np.abs(S1[0]) ** 2)


def test_smatrix_two_particle_non_interacting():
    m = scattering.Model([0] * 2, [[0, 1, 1]], [0] * 2)
    channels = [
        scattering.Channel(site=0, strength=1),
        scattering.Channel(site=1, strength=1),
    ]
    s = scattering.Setup(m, channels)

    E = 0
    dE = 0
    chls = [[0, 0], [0, 1], [1, 0], [1, 1]]

    S2s = np.zeros((4, 4), np.float64)
    for ci, chlsi in enumerate(chls):
        for co, chlso in enumerate(chls):
            _, _, S2, _ = smatrix.two_particle(s,
                                               chlsi=chlsi,
                                               chlso=chlso,
                                               E=E,
                                               dE=dE)

            S2s[ci, co] = np.sum(np.abs(S2) ** 2)

    # Test if S2 vanishes
    assert np.allclose(np.ones((4, 4)) + S2s, np.ones((4, 4)), 1e-2), \
        'S2 does not vanish for non-interacting systems, S2 = {0}.'.format(np.abs(S2s))


def test_smatrix_optical_theorem(verbose=False):
     # setup
    # m = scattering.Model(omegas=[0, 0], links=[[0, 1, 1.]], U=0.)

    for chlsi in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        domega, dE = np.random.rand(2)
        _test_smatrix_optical_theorem(5 * domega, 5 * dE, chlsi, verbose=verbose)


def _test_smatrix_optical_theorem2(chlsi):
    m = scattering.Model(
        omegas=[0] * 2,
        links=[[0, 1, .5]],
        U=[5] * 2)

    channels = []
    channels.append(scattering.Channel(site=0, strength=1))
    channels.append(scattering.Channel(site=1, strength=1))

    s = scattering.Setup(m, channels)

    # input statei
    E = 0
    dE = 0
    chlsos = [[0, 0], [0, 1], [1, 0], [1, 1]]

    qs = np.linspace(-100, 100, 8192, endpoint=False)

    D2s, S2s = [], []
    for c, chlso in enumerate(chlsos):
        qs, D2, S2, S2in = smatrix.two_particle(s,
                                                chlsi=chlsi,
                                                chlso=chlso,
                                                E=E,
                                                dE=dE,
                                                qs=qs)

        D2s.append(D2)
        S2s.append(S2)

    iq0 = np.where(qs == E / 2)[0][0]

    dsums = [np.sum(np.abs(D2s[c]) ** 2) for c in xrange(4)]
    ssums = [np.sum(np.abs(S2s[c]) ** 2) * (qs[1] - qs[0]) for c in xrange(4)]
    dssums = [2 * np.real(np.sum(D2s[c]).conj() * S2s[c][iq0]) for c in xrange(4)]

    # Test single particle contribution
    assert np.isclose(np.sum(dsums), 2, 1e-3), \
        'S(1)S(1) is not normalized (= {0})'.format(np.sum(dsums))

    # Test optical theorem
    assert np.isclose(np.sum(ssums), -np.sum(dssums), 1e-2), \
        'Optical theorem broken, S2.S2 = {0} != -S1.S2 = {1}.'.format(np.sum(ssums), -np.sum(dssums))


def _test_smatrix_optical_theorem(domega, dE, chlsi, verbose=False):

    E = 0

    m = scattering.Model(
        omegas=[0] * 2,
        links=[[0, 1, .5]],
        U=[5] * 2)

    channels = []
    channels.append(scattering.Channel(site=0, strength=1))
    channels.append(scattering.Channel(site=1, strength=1))

    s = scattering.Setup(m, channels)

    # input statei
    chlsos = [[0, 0], [0, 1], [1, 0], [1, 1]]

    d = 100
    # 50 / np.max(np.abs(dE), .5)
    print(d)
    qs, iq0 = scattering.energies(E, dE, N=8192, WE=2 * d)

    D2s, S2s = [], []
    for c, chlso in enumerate(chlsos):
        qs, D2, S2, S2in = smatrix.two_particle(s,
                                                chlsi=chlsi,
                                                chlso=chlso,
                                                E=E,
                                                dE=dE,
                                                qs=qs
                                                )

        D2s.append(D2)
        S2s.append(S2)

    # iq0s = [np.where(np.isclose(qs, (E - dE) / 2))[0][0], np.where(np.isclose(qs, (E + dE) / 2))[0][0]]

    dsums = [.5 * np.sum(np.abs(D2s[c]) ** 2) for c in xrange(4)]
    ssums = [.5 * np.sum(np.abs(S2s[c]) ** 2) * (qs[1] - qs[0]) for c in xrange(4)]
    dssums = [.5 * 2 * np.real(np.sum(D2s[c]).conj() * S2s[c][iq0]) for c in xrange(4)]

    # dsums = [.5 * np.sum(np.abs(D2s[c]) ** 2) for c in xrange(4)]
    # ssums = [.5 * np.sum(np.abs(S2s[c]) ** 2) * (qs[1] - qs[0]) for c in xrange(4)]
    # dssums = [.5 * 2 * np.real(np.sum(D2s[c].conj() * S2s[c][iq0])) for c in xrange(4)]

    if verbose:
        print('For d(2) = {} and dE = {}:'.format(domega, dE))

    # Test single particle contribution
    assert np.isclose(np.sum(dsums), 1, 1e-3), \
        'S(1)S(1) is not normalized (= {0})'.format(np.sum(dsums))

    if verbose:
        print('S1.S1 norm is: {0}'.format(np.sum(dsums)))

    # Test optical theorem
    assert np.isclose(np.sum(ssums), -np.sum(dssums), 1e-2), \
        'Optical theorem broken for parameters: E={0}, dE={1}, chlsi={2}.'.format(E, dE, chlsi) + \
        '\n' + \
        'Compare S2.S2 = {0} != -S1.S2 = {1}.'.format(np.sum(ssums), -np.sum(dssums))

    if verbose:
        print('Compare S2.S2 = {0} to -S1.S2 = {1}.'.format(np.sum(ssums), -np.sum(dssums)))


if __name__ == '__main__':

    test_smatrix_optical_theorem(verbose=True)
