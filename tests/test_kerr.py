from __future__ import division, print_function
import numpy as np
import scattering
import g2


def kerr(E=0, U=0, gs=None):
    """
    Setup the Kerr nonlinear element
    """
    model = scattering.Model(
        omegas=[E]*1,
        links=[],
        U=[U])

    if gs is None:
        gs = (0.1, 0.1)

    channels = []
    channels.append(scattering.Channel(site=0, strength=gs[0]))
    channels.append(scattering.Channel(site=0, strength=gs[1]))

    se = scattering.Setup(model, channels)
    se.label = 'U={0}'.format(U)

    return se


def test_compare_kerr_g2_numerics_with_analytics():
    """
    Test our numerical calcalulation against analytical results
    """
    U = 5*np.random.rand(1)[0]
    g = np.random.rand(1)[0]
    se = kerr(0, U, (g, g))

    deltas = np.linspace(0, 2, 64)
    taus = np.linspace(0, 20, 64)

    # numerics
    g2n_11 = np.zeros((len(deltas), len(taus)))
    g2n_22 = np.zeros((len(deltas), len(taus)))
    for i, delta in enumerate(deltas):
        g2n_11[i, :] = g2.coherent_state(se, (0, 0), (0, 0), 2*delta, taus)[0]
        g2n_22[i, :] = g2.coherent_state(se, (0, 0), (1, 1), 2*delta, taus)[0]

    # analytics
    gamma = 2*np.pi*g**2  # coupling gamma
    g2a_11, g2a_22 = g2_analytic(gamma, U, deltas, taus)

    assert np.allclose(g2n_11, g2a_11)
    assert np.allclose(g2n_22, g2a_22)


def g2_analytic(gamma, U, deltas, taus):
    """
    g2 for weakly coherent light as derived by Mikhail.
    """
    g2_11 = np.zeros((len(deltas), len(taus)))
    g2_22 = np.zeros((len(deltas), len(taus)))

    for i, delta in enumerate(deltas):
        g2_11[i, :] = (gamma/(delta+1e-12))**4*np.abs((delta/gamma)**2 +
                                                      np.exp(1j*(delta+1j*gamma)*taus) -
                                                      (delta + 1j*gamma)/(delta + 1j*gamma - U/2)*np.exp(1j*(delta + 1j*gamma)*taus))**2

        g2_22[i, :] = np.abs(1 -
                             np.exp(1j*(delta + 1j*gamma)*taus) +
                             (delta + 1j*gamma)/(delta + 1j*gamma - U/2)*np.exp(1j*(delta + 1j*gamma)*taus))**2

    return g2_11, g2_22
