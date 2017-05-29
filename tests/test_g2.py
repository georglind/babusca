from __future__ import division, print_function
import sys
sys.path.append('../')
import numpy as np
import scattering
import g2

import matplotlib.pyplot as plt
# import seaborn
# seaborn.set()


def test_g2_unity_in_noninteracting_system():
    # model with two sites and no interactiong
    N, U = 2, 0

    model = scattering.Model(
        omegas=[0]*N,
        links=[[i, i+1, 1] for i in xrange(N-1)],
        U=[2*U]*N)

    # channels
    channels = []
    channels.append(scattering.Channel(site=0, strength=.4))
    channels.append(scattering.Channel(site=N-1, strength=1.1))

    setup = scattering.Setup(model, channels)

    Es = np.linspace(-6, 20, 128)
    chlso = [1, 1]

    g12 = np.zeros((len(Es),), dtype=np.float64)
    g22 = np.zeros((len(Es),), dtype=np.float64)
    for i, E in enumerate(Es):
        g12[i] = np.real(g2.coherent_state_tau0(setup, [0, 0], chlso, E)[0][0])
        g22[i] = np.real(g2.general_state(setup, [0, 0], chlso, E, 0, 0)[0])

    assert np.allclose(g22, 1), \
        'g2 != 1 for a weakly coherent state in a non-interacting system'

    assert np.allclose(g22, 1), \
        'g2 != 1 for a general two-photon state in a non-interacting system'


def test_g2_optical_theorem():
    Ns = [2, 4]
    Us = [1, 2, 4]

    for N in Ns:
        for U in Us:
            gs = 5*np.random.rand(2) + .1
            _test_g2_optical_theorem(N, U, gs)


def _test_g2_optical_theorem(N, U, gs=(1, 1)):
    N, U, = 2, 2

    model = scattering.Model(
        omegas=[0]*N,
        links=[(0, 1, 1)],
        U=[2*U]*N)

    channels = []
    channels.append(scattering.Channel(site=0, strength=gs[0]))
    channels.append(scattering.Channel(site=N-1, strength=gs[1]))

    setup = scattering.Setup(model, channels)

    # Es = np.linspace(-6, 20, 128)
    E = 10
    chlsi = (0, 0)
    dE = .1

    i1, i2 = chlsi
    k0, q0 = .5*E, .5*dE

    E1 = setup.eigenenergies(1)

    ps = []
    nchan = len(setup.channels)
    for i in xrange(nchan):
        pj = []
        for j in xrange(nchan):
            pj.append(g2.phi(setup, k0, q0, i, j, i1, i2))
        ps.append(pj)

    # fij2_sqrd
    fij2_sqrd = 0
    for i, _ in enumerate(setup.channels):
        for j, _ in enumerate(setup.channels):
            for ll in xrange(nchan):
                for lp in xrange(nchan):
                    fij2_sqrd += (ps[i][j][lp].conj()*ps[i][j][ll]
                                  + ps[j][i][lp].conj()*ps[j][i][ll])/(E1[lp].conj() - E1[ll])

    fij2_sqrd *= 2*np.pi*1j
    # print(np.real(fij2_sqrd[0]))

    # croff
    croff = 0
    for i in xrange(nchan):
        for j in xrange(nchan):
            croff += g2.fijm(setup, k0, q0, i, j, i1, i2).conj()*g2.fij2(k0 + q0, k0 - q0, ps[i][j], ps[j][i], E1) \
                + g2.fijp(setup, k0, q0, i, j, i1, i2).conj()*g2.fij2(k0 - q0, k0 + q0, ps[i][j], ps[j][i], E1)

    croff = 2*np.real(croff)
    # print(croff[0])

    assert np.isclose(np.real(fij2_sqrd[0]), -np.real(croff[0]), 1e-6), \
        'Optical theorem broken since: {0} != {1}'.format(np.real(fij2_sqrd[0]), croff[0])


def test_g2_coherent_tau0_and_g2_coherent():
    N, U, = 2, 2
    gs = (0.1, 0.1)

    model = scattering.Model(
        omegas=[0]*N,
        links=[(0, 1, 1)],
        U=[2*U]*N)

    channels = []
    channels.append(scattering.Channel(site=0, strength=gs[0]))
    channels.append(scattering.Channel(site=N-1, strength=gs[1]))

    setup = scattering.Setup(model, channels)

    Es = np.linspace(-3, 12, 256)

    g2stau0 = np.zeros((len(Es),), dtype=np.float64)
    g2s = np.zeros((len(Es),), dtype=np.float64)
    for i, E in enumerate(Es):
        g2stau0[i] = g2.coherent_state_tau0(setup, (0, 0), (1, 1), E)[0]
        g2s[i] = g2.coherent_state(setup, (0, 0), (1, 1), E, 0)[0]

    assert np.allclose(g2stau0, g2s), \
        'g2_coherent_tau0 and g2_coherent do not coincide at tau=0'


def test_g2_fock_state():

    N, U, = 2, 0
    gs = (.2, .1)

    model = scattering.Model(
        omegas=[0]*N,
        links=[(0, 1, 1)],
        U=[2*U]*N)

    channels = []
    channels.append(scattering.Channel(site=0, strength=gs[0]))
    channels.append(scattering.Channel(site=N-1, strength=gs[1]))

    setup = scattering.Setup(model, channels)

    Es = np.linspace(-3, 12, 1024)
    dE = 0

    g2s = np.zeros(Es.shape, dtype=np.complex128)
    g2n = np.zeros(Es.shape, dtype=np.complex128)
    g2d = np.zeros(Es.shape, dtype=np.complex128)

    for i, E in enumerate(Es):
        g2s[i], g2n[i], g2d[i] = g2.fock_state(setup, (0, 0), (1, 1), E, dE)

    plt.semilogy(Es, g2s, label='g2')
    plt.semilogy(Es, g2n, label='g2n')
    plt.semilogy(Es, g2d, label='g1g1')
    plt.legend()
    plt.show()

def compare_g2_fock_state_to_g2_coherent_state():

    N, U, = 2, 0
    gs = (.2, .1)

    model = scattering.Model(
        omegas=[0]*N,
        links=[(0, 1, 1)],
        U=[2*U]*N)

    channels = []
    channels.append(scattering.Channel(site=0, strength=gs[0]))
    channels.append(scattering.Channel(site=N-1, strength=gs[1]))

    setup = scattering.Setup(model, channels)

    Es = np.linspace(-3, 12, 1024)
    dE = 0

    g2f = np.zeros(Es.shape, dtype=np.complex128)
    g2c = np.zeros(Es.shape, dtype=np.complex128)

    for i, E in enumerate(Es):
        g2s[i], _, _ = g2.fock_state(setup, (0, 0), (1, 1), E, dE)
    

if __name__ == '__main__':

    # all tests
    # test_g2_unity_in_noninteracting_system
    # test_g2_optical_theorem()
    # test_g2_coherent_tau0_and_g2_coherent()
    test_g2_fock_state()
