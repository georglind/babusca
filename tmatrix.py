from __future__ import division, print_function
import numpy as np
import scattering

#     $$$$$$$$\                               $$\               $$\
#     \__$$  __|                              $$ |              \__|
#        $$ |        $$$$$$\$$$$\   $$$$$$\ $$$$$$\    $$$$$$\  $$\ $$\   $$\
#        $$ |$$$$$$\ $$  _$$  _$$\  \____$$\\_$$  _|  $$  __$$\ $$ |\$$\ $$  |
#        $$ |\______|$$ / $$ / $$ | $$$$$$$ | $$ |    $$ |  \__|$$ | \$$$$  /
#        $$ |        $$ | $$ | $$ |$$  __$$ | $$ |$$\ $$ |      $$ | $$  $$<
#        $$ |        $$ | $$ | $$ |\$$$$$$$ | \$$$$  |$$ |      $$ |$$  /\$$\
#        \__|        \__| \__| \__| \_______|  \____/ \__|      \__|\__/  \__|


def single_particle(sp, chli, chlo, Es=None):
    """alias for one_particle"""
    return one_particle(sp, chli, chlo, Es)


def single_photon(sp, chli, chlo, Es=None):
    """alias for one_particle"""
    return one_particle(sp, chli, chlo, Es)


def one_photon(sp, chli, chlo, Es=None):
    """alias for one_particle"""
    return one_particle(sp, chli, chlo, Es)


def one_particle(sp, chli, chlo, Es=None):
    if (sp.local):
        return one_particle_local(sp, chli, chlo, Es)
    else:
        return one_particle_quasilocal(sp, chli, chlo, Es)


def one_particle_local(sp, chli, chlo, Es=None):
    """
    Calculate the one-particle irreducible T-matrix T(1).

    Parameters
    ----------
    s : Setup
        Setup object describing the setup
    chi : int
        Input schannel
    cho : int
        Output channel
    Es : ndarray
        List of particle energies
    """
    E1, _, _ = sp.eigenbasis(1)

    # numerators
    num1 = sp.transition(0, chli, 1)
    num2 = sp.transition(1, chlo, 0)

    # guess a suitable range of energies to probe
    if Es is None:
        Es = np.linspace(-np.max(np.abs(E1)) - 1, np.max(np.abs(E1)) + 1, 1000)

    # initialize the matrix
    T1 = np.zeros((Es.shape), dtype=np.complex128)

    # removing copling contants from setup
    # num = sp.gs[chli] * sp.gs[chlo] * num2.T * num1
    num = num2.T * num1

    for k in xrange(len(E1)):
        T1[:] += num[k] / (Es - E1[k])

    return Es, T1


def one_particle_quasilocal(sp, chli, chlo, Es=None):
    """
    Calculate the one-particle irreducible T-matrix T(1).

    Parameters
    ----------
    s : Setup
        Setup object describing the setup
    chi : int
        Input schannel
    cho : int
        Output channel
    Es : ndarray
        List of particle energies
    """
    T1 = np.zeros((Es.shape), dtype=np.complex128)

     # guess a suitable range of energies to probe
    if Es is None:
        maxt = np.max(np.abs(sp.model.links)) + np.max(np.abs(sp.model.omegas)) + 1
        Es = np.linspace(- maxt, maxt, 1000)

    for i, E in enumerate(Es):
        # single particle eigenenergies
        E1, _, _ = sp.eigenbasis(1, E)

        # numerators
        num1 = sp.transition(0, chli, 1, E)
        num2 = sp.transition(1, chlo, 0, E)

        # initialize the matrix
        # num = sp.gs[chli] * sp.gs[chlo] * num2.T * num1
        num = num2.T * num1

        for k in xrange(len(E1)):
            T1[i] += num[k] / (E - E1[k])

    return Es, T1


# only local implementation

def two_particle_local(se, chlsi, chlso, E, dE, qs=None, verbal=False):
    """
    Calculate the two-particle irreducible T-matrix T(2).

    Parameters
    ----------
    se : Setup object
        Describes the Scattering setup
    chlsi : list
        Input state channels (in) [k0, k1]
    chlso : list
        Output state channels (out) [k'0, k'1]
    E : float
        Total two-particle energy
    dE : float
        Energy difference between the two input particles
    qs : List of scattering energies.
    """
    # Energies of the incoming bosons
        # Comparing to Mikhails notes:
        # nu_1 = E - q
        # nu_2 = q
        # nu_3 = .5(E - dE)
        # nu_4 = .5(E + dE)
    # energy of two incoming photons
    nu0 = 0.5 * (E - dE)
    nu1 = 0.5 * (E + dE)

    # creation/annihilation operator in n = 0-1 eigenbasis representation
    A10 = [se.transition(0, chl, 1) for chl in chlsi]
    A01 = [se.transition(1, chl, 0) for chl in chlso]

    # single particle sector eigenbasis
    E1, _, _ = se.eigenbasis(1)

    # creation/annihilation operator in n = 1-2 eigenbasis representation
    A21 = [se.transition(1, chl, 2) for chl in chlsi]
    A12 = [se.transition(2, chl, 1) for chl in chlso]

    # two-particle sector eigenbasis
    E2, _, _ = se.eigenbasis(2)

    if verbal:
        print('tmatrix')
        print(A10[0])
        print(A21[0])
        print(E2)

    # the interactive part
    op0123 = (A12[1]
              .dot(np.diag(1 / (E - E2)))
              .dot(A21[0])
              .dot(A10[1] / (nu1 - E1[:, None])))

    op0132 = (A12[1]
              .dot(np.diag(1 / (E - E2)))
              .dot(A21[1])
              .dot(A10[0] / (nu0 - E1[:, None])))

    # add those two many-body routes
    op01aa = op0123 + op0132

    if verbal:
        print('A')
        print(op0123)

    op1023 = (A12[0]
              .dot(np.diag(1 / (E - E2)))
              .dot(A21[0])
              .dot(A10[1] / (nu1 - E1[:, None])))

    op1032 = (A12[0]
              .dot(np.diag(1 / (E - E2)))
              .dot(A21[1])
              .dot(A10[0] / (nu0 - E1[:, None])))

    # add those two many-body routes
    op10aa = op1023 + op1032

    # reshape E1 vector
    E1p = E1[:, None]

    # single particle contributions
    op00 = A01[0] * A10[0].T
    op01 = A01[0].T * A10[1]
    op10 = A01[1] * A10[0].T
    op11 = A01[1].T * A10[1]
    opE = (E - E1p - E1p.T) / ((nu0 - E1p) * (nu1 - E1p.T))

    # Output energies for one particle
    if qs is None:
        qs = scattering.energies(E, dE)

    # init matrices
    T2s = np.zeros((len(qs), 1), dtype=np.complex128)
    N2s = np.zeros((len(qs), 1), dtype=np.complex128)

    # calculate for each possible outgoing energy combination
    for iq, q in enumerate(qs):
        # energy of remaining outgoing particle
        Eq = E - q

        T2s[iq] = ((A01[0] / (q - E1[None, :])).dot(op01aa))[0]
        T2s[iq] += ((A01[1] / (Eq - E1[None, :])).dot(op10aa))[0]

        # The non-interactive part
        opni = opE / ((q - E1p) * (Eq - E1p.T))
        opnis = opE / ((Eq - E1p) * (q - E1p.T))

        N2s[iq] = -op00.dot(opni).dot(op11)[0][0]
        N2s[iq] -= op10.dot(opnis).dot(op01)[0][0]

    prefactor = 0.25

    # finalize
    T2s *= prefactor
    N2s *= prefactor

    return qs, T2s + N2s, N2s
