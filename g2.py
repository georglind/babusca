from __future__ import division, print_function
import numpy as np
import scattering
import smatrix
import utilities as util


def coherent_state_tau0(setup, chlsi, chlso, E=0):
    """
    g2 for no delay.

    Parameters
    ----------
    se : Setup
        Scattering setup object
    chlsi : list
        list of incoming channels (must be identical!)
    chlso : list
        list of outgoing channels
    E : float or list
        Incoming two photon state energ(y/ies)
    """
    if setup.local:
        return coherent_state_tau0_local(setup, chlsi, chlso, E)
    else:
        return coherent_state_tau0_quasilocal(setup, chlsi, chlso, E)


def coherent_state_tau0_local(setup, chlsi, chlso, E=0):
    """
    g2 from Mikhails formula

    Parameters
    ----------
    se : Setup
        Scattering setup object
    chlsi : list
        list of incoming channels (must be identical!)
    chlso : list
        list of outgoing channels
    E : float or list
        Incoming two photon state energ(y/ies)
    """
    (i1, i2) = chlsi
    (o1, o2) = chlso

    assert i1 == i2, 'Photons in an incoming coherent state must all belong to the same channel.'

    # convert to numpy array
    k0s = np.atleast_1d(E) / 2

    _, S00 = smatrix.one_particle(setup, i1, o1, k0s)
    _, S11 = smatrix.one_particle(setup, i2, o2, k0s)

    g2a = 1 - (S11 - util.delta(i2, o2)) * (S00 - util.delta(i1, o1)) / (S11 * S00)

    # eigen energies
    E1 = setup.eigenenergies(1)
    E2 = setup.eigenenergies(2)

    # creation operator in n = 0-1 eigenbasis representation
    A01 = setup.transition(1, o1, 0)
    A12 = setup.transition(2, o2, 1)
    A21 = setup.transition(1, i1, 2)
    A10 = setup.transition(0, i2, 1)

    g2b = np.zeros(k0s.shape, np.complex128)
    for i, k0 in enumerate(k0s):
        g2b[i] = A01 \
            .dot(A12) \
            .dot(np.diag(1 / (2 * k0 - E2))) \
            .dot(A21) \
            .dot(A10 / (k0 - E1[:, None]))[0][0]

    prefactor = 4 * np.pi ** 2  # * np.prod([setup.gs[ch] for ch in chlsi + chlso])

    return {'g2': np.abs(g2a - prefactor * g2b / (S11 * S00)) ** 2,
            'phi2': np.abs(g2a * S11 * S00 - prefactor * g2b) ** 2,
            'normalization': S11 * S00,
            'phi2_reducible': g2a * S11 * S00,
            'phi2_irreducible': - prefactor * g2b}


def coherent_state_tau0_quasilocal(setup, chlsi, chlso, E=0):
    """
    g2 from Mikhails formula

    Parameters
    ----------
    se : Setup
        Scattering setup object
    chlsi : list
        list of incoming channels (must be identical!)
    chlso : list
        list of outgoing channels
    E : float or list
        Incoming two photon state energ(y/ies)
    """
    (i1, i2) = chlsi
    (o1, o2) = chlso

    assert i1 == i2, 'Photons in an incoming coherent state **must** all belong to the same channel.'

    # convert to numpy array
    k0s = np.atleast_1d(E) / 2
    # k0s = E / 2 if util.isarray(E) else np.array([E / 2])

    _, S00 = smatrix.one_particle(setup, i1, o1, k0s)
    _, S11 = smatrix.one_particle(setup, i2, o2, k0s)

    g2a = 1 - (S11 - (i2 == o2)) * (S00 - (i1 == o1)) / (S11 * S00)
    g2b = np.zeros(k0s.shape, np.complex128)

    for i, k0 in enumerate(k0s):
        setup.reset()  # remove cache

        # eigen-energies
        E1 = setup.eigenenergies(1, phi=k0)
        E2 = setup.eigenenergies(2, phi=k0)

        # creation operator in n = 0-1-2 eigenbasis representation
        A01 = setup.transition(1, o1, 0, k0)
        A12 = setup.transition(2, o2, 1, k0)
        A21 = setup.transition(1, i1, 2, k0)
        A10 = setup.transition(0, i2, 1, k0)

        # g2
        g2b[i] = A01 \
            .dot(A12) \
            .dot(np.diag(1 / (2 * k0 - E2))) \
            .dot(A21) \
            .dot(A10 / (k0 - E1[:, None]))[0][0]

    prefactor = 4 * np.pi ** 2   # * np.prod([setup.gs[ch] for ch in chlsi + chlso])

    return {'g2': np.abs(g2a - prefactor / (S11 * S00) * g2b) ** 2,
            'phi2': np.abs(g2a * S11 * S00 - prefactor * g2b) ** 2,
            'normalization': S11 * S00,
            'phi2_reducible': g2a * S11 * S00,
            'phi2_irreducible': - prefactor * g2b}


def coherent_state(setup, chlsi, chlso, E=0, tau=0, verbose=False):
    """
    g2(E, tau) correlation as a function of incoming energy, E, and
    delay time, tau.

    Parameters
    ----------
    se : Setup
        Scattering setup object
    chlsi : list
        list of incoming channels (must be identical!)
    chlso : list
        list of outgoing channels
    E : float or list
        Incoming two photon state energ(y/ies)
    """
    if setup.local:
        return coherent_state_local(setup, chlsi, chlso, E, tau, verbose=verbose)
    else:
        return coherent_state_quasilocal(setup, chlsi, chlso, E, tau)


def coherent_state_local(setup, chlsi, chlso, E=0, tau=0, verbose=False):
    """
     g2(E, tau) correlation as a function of incoming energy, E, and
    delay time, tau. Here specifically for local systems.

    Parameters
    ----------
    se : Setup
        Scattering setup object
    chlsi : list
        list of incoming channels (must be identical!)
    chlso : list
        list of outgoing channels
    E : float or list
        Two photon state energ(y/ies)
    """
    # channels
    # two incoming photons from same channel
    assert chlsi[0] == chlsi[1], 'Photons in an incoming coherent state must all belong to the same channel.'

    (i1, _) = chlsi
    (o1, o2) = chlso

    # k0s
    k0s = np.atleast_1d(E) / 2
    # k0s = E / 2 if util.isarray(E) else np.array([E / 2])

    # numpy array
    taus = np.atleast_1d(tau)

    # smatrix single photon
    _, S00 = smatrix.one_particle(setup, i1, o1, k0s)
    _, S11 = smatrix.one_particle(setup, i1, o2, k0s)

    # eigen energies
    E1 = setup.eigenenergies(1)
    E2 = setup.eigenenergies(2)

    # creation operator in n = 0-1 eigenbasis representation
    A01_i = setup.transition(1, o1, 0)
    A12_j = setup.transition(2, o2, 1)
    A01_j = setup.transition(1, o2, 0)
    A21_0 = setup.transition(1, i1, 2)
    A10_0 = setup.transition(0, i1, 1)

    prefactor = 4 * np.pi ** 2   # * np.prod([setup.gs[c] for c in chlsi + chlso])

    g2 = np.zeros((len(k0s), len(taus)), dtype=np.float64)
    S2 = np.zeros((len(k0s), len(taus)), dtype=np.complex128)

    for i, k0 in enumerate(k0s):

        op = (A12_j.dot(np.diag(1 / (2 * k0 - E2))).dot(A21_0)
              - np.diag(1 / (k0 - E1)).dot(A10_0).dot(A01_j)) \
            .dot(A10_0 / (k0 - E1[:, None]))

        S2[i, :] = prefactor * np.dot(A01_i * np.exp(-1j * (E1[None, :] - k0) * taus[:, None]), op)[:, 0]
        g2[i, :] = np.abs(1 - S2[i, :] / (S00[i] * S11[i])) ** 2

            # np.abs(1 - (prefactor/(S00[i]*S11[i])
                        # * np.dot(A01_i*np.exp(-1j*(E1[None, :] - k0)*taus[:, None]), op))
                   # )[:, 0]**2

    return {'g2': g2,
            'phi2': np.abs(np.tile(S00 * S11, (len(taus), 1)) - S2) ** 2,
            'normalization': S11 * S00,
            'phi2_irreducible': S2}


def coherent_state_quasilocal(setup, chlsi, chlso, E=0, tau=0):
    """
    g2(E, tau) correlation as a function of incoming energy, E, and
    delay time, tau. Here specifically for quasi-local systems.

    Parameters
    ----------
    se : Setup
        Scattering setup object
    chlsi : list
        list of incoming channels (must be identical!)
    chlso : list
        list of outgoing channels
    E : float or list
        Two photon state energ(y/ies)
    """
    # channels
    # two incoming photons from same channel
    assert chlsi[0] == chlsi[1], 'Photons in an incoming coherent state must all belong to the same channel.'

    (i1, _) = chlsi
    (o1, o2) = chlso

    # numpy arraify
    k0s, taus = np.atleast_1d(E / 2, tau)

    # smatrix single photon
    _, S00 = smatrix.one_particle(setup, i1, o1, k0s)
    _, S11 = smatrix.one_particle(setup, i1, o2, k0s)

    g2 = np.zeros((len(k0s), len(taus)), dtype=np.float64)
    S2 = np.zeros((len(k0s), len(taus)), dtype=np.complex128)

    for i, k0 in enumerate(k0s):

        # eigen energies
        E1 = setup.eigenenergies(1, phi=k0)
        E2 = setup.eigenenergies(2, phi=k0)

        # creation operator in n = 0-1 eigenbasis representation
        A01_i = setup.transition(1, o1, 0, k0)
        A12_j = setup.transition(2, o2, 1, k0)
        A01_j = setup.transition(1, o2, 0, k0)

        A21_0 = setup.transition(1, i1, 2, k0)
        A10_0 = setup.transition(0, i1, 1, k0)

        prefactor = 4 * np.pi ** 2 * np.prod([setup.gs[c] for c in chlsi + chlso])

        op = (A12_j.dot(np.diag(1 / (2 * k0 - E2))).dot(A21_0)
              - np.diag(1 / (k0 - E1)).dot(A10_0).dot(A01_j)) \
            .dot(A10_0 / (k0 - E1[:, None]))

        S2[i, :] = prefactor * np.dot(A01_i * np.exp(-1j * (E1[None, :] - k0) * taus[:, None]), op)[:, 0]
        g2[i, :] = np.abs(1 - S2[i, :] / (S00[i] * S11[i])) ** 2

        # np.abs(1 - (prefactor/(S00[i]*S11[i])
                        # * np.dot(A01_i*np.exp(-1j*(E1[None, :] - k0)*taus[:, None]), op))
                   # )[:, 0]**2

    return {'g2': g2,
            'phi2': np.abs(np.tile(S00 * S11, (len(taus), 1)) - S2) ** 2,
            'normalization': S11 * S00,
            'phi2_irreducible': S2}

    # return g2, np.abs(np.tile(S00 * S11, (len(taus), 1)) - S2) ** 2, S2


def fock_state_local(setup, chlsi, chlso, E, dE, tau=0):
    """
    Compute g^2 correlation function for a general two-photon state.

    Parameters
    ----------
    se : scattering.Model
        A given scattering model
    chlsi : list
        List of incoming two-photon state channel indices.
    chlso : list
        List of outgoing two-photon state channel indices.
    E : float
        Total two-photon energy
    dE : float
        Energy difference between the two photons.
    tau : float
        Time
    """
    # channel indices
    (i1, i2) = chlsi
    (o1, o2) = chlso

    # single photon energies
    k0, q0 = .5 * E, .5 * dE
    k1, k2 = k0 + q0, k0 - q0

    # times
    taus = np.atleast_1d(tau)

    _, S11 = smatrix.one_particle(setup, i1, o1, np.array([k1]))
    _, S22 = smatrix.one_particle(setup, i2, o2, np.array([k2]))

    _, S12 = smatrix.one_particle(setup, i1, o2, np.array([k1]))
    _, S21 = smatrix.one_particle(setup, i2, o1, np.array([k2]))

    g2a = S11 * S22 * np.exp(-1j * q0 * taus) + S21 * S12 * np.exp(1j * q0 * taus)

    # eigen energies
    E1, E2 = [setup.eigenenergies(nb) for nb in (1, 2)]

    # creation operator in n = 0-1 eigenbasis representation
    A10, A01, A21, A12 = transitions(setup)

    prefactor = (2 * np.pi) ** 2 * np.prod([setup.gs[c] for c in chlsi + chlso])

    At = A01[o1] * np.exp(-1j * (E1[None, :] - k0) * taus[:, None])

    g2b = At.dot(A12[o2]).dot(np.diag(1 / (E - E2))).dot(A21[i2]).dot(A10[i1] / (k1 - E1[:, None])) \
        + At.dot(A12[o2]).dot(np.diag(1 / (E - E2))).dot(A21[i1]).dot(A10[i2] / (k2 - E1[:, None])) \
        - (At / (k2 - E1)).dot(A10[i2]).dot(A01[o2]).dot(A10[i1] / (k1 - E1[:, None])) \
        - (At / (k1 - E1)).dot(A10[i1]).dot(A01[o2]).dot(A10[i2] / (k2 - E1[:, None]))

    g2 = np.abs(g2a - prefactor * g2b.T)[:][0] ** 2
    # normalization
    if o2 == o1:
        g1ig1j = g1(setup, chlsi, k0, q0, o1) ** 2
    else:
        g1ig1j = g1(setup, chlsi, k0, q0, o1) * g1(setup, chlsi, k0, q0, o2)

    # return
    return g2 / np.abs(g1ig1j[0]), g2, g1ig1j[0]


def transitions(setup):
    A01 = [setup.transition(1, i, 0) for i, chl in enumerate(setup.channels)]
    A12 = [setup.transition(2, i, 1) for i, chl in enumerate(setup.channels)]
    A21 = [setup.transition(1, i, 2) for i, chl in enumerate(setup.channels)]
    A10 = [setup.transition(0, i, 1) for i, chl in enumerate(setup.channels)]

    return [A10, A01, A21, A12]


def g1(setup, chlsi, k0, q0, i):

    (i1, i2) = chlsi
    E1 = setup.eigenenergies(1)

    _, Sk1 = smatrix.one_particle(setup, i1, i, np.array([k0 + q0]))
    _, Sk2 = smatrix.one_particle(setup, i2, i, np.array([k0 - q0]))

    fij_sqrd = np.abs(Sk1) ** 2 + np.abs(Sk2) ** 2

    fij2_sqrd = 0
    # EE = E1[None, :].conj() - E1[:, None]

    pp, pm = [], []
    nchan = len(setup.channels)
    for j in xrange(nchan):
        pp.append(phi(setup, k0, q0, i, j, i1, i2))
        pm.append(phi(setup, k0, q0, j, i, i1, i2))

        # fij2_sqrd += np.sum((pp[j].conj().T*pp[j] + pm[j].conj().T*pm[j])/EE)

        for ll in xrange(len(E1)):
            for lp in xrange(len(E1)):
                fij2_sqrd += (pp[j][lp].conj() * pp[j][ll]
                              + pm[j][lp].conj() * pm[j][ll]) / (E1[lp].conj() - E1[ll])

    # add a prefactor
    fij2_sqrd *= 2 * np.pi * 1j

    croff = 0
    for j in xrange(len(setup.channels)):
        croff += fijm(setup, k0, q0, i, j, i1, i2).conj() * fij2(k0 + q0, k0 - q0, pp[j], pm[j], E1) \
            + fijp(setup, k0, q0, i, j, i1, i2).conj() * fij2(k0 - q0, k0 + q0, pp[j], pm[j], E1)

    return fij_sqrd + fij2_sqrd + 2 * np.real(croff)


def phi(setup, k0, q0, i, j, i1, i2):

    E1 = setup.eigenenergies(1)[:, None]
    E2 = setup.eigenenergies(2)

    A10, A01, A21, A12 = transitions(setup)

    return -2 * np.pi * 1j * np.prod([setup.gs[c] for c in [i, j, i1, i2]]) * \
        A01[i].T * (
            (
                A12[j]
                .dot(np.diag(1 / (2 * k0 - E2)))
                .dot(A21[i2])
                -
                (A10[i2] / (k0 - q0 - E1)).dot(A01[j])
            ).dot(
                A10[i1] / (k0 + q0 - E1)
            )
            +
            (
                A12[j]
                .dot(np.diag(1 / (2 * k0 - E2)))
                .dot(A21[i1])
                -
                (A10[i1] / (k0 + q0 - E1)).dot(A01[j])
            ).dot(
                A10[i2] / (k0 - q0 - E1)
            )
        )


def fij2(enp, enm, phip, phim, E1):
    return np.sum(phip / (enp - E1[:, None])) + np.sum(phim / (enm - E1[:, None]))


def fijm(setup, k0, q0, i, j, i1, i2):
    _, Si = smatrix.one_particle(setup, i1, i, np.array([k0 + q0]))
    _, Sj = smatrix.one_particle(setup, i2, j, np.array([k0 - q0]))

    return Si * Sj


def fijp(setup, k0, q0, i, j, i1, i2):
    _, Si = smatrix.one_particle(setup, i1, j, np.array([k0 + q0]))
    _, Sj = smatrix.one_particle(setup, i2, i, np.array([k0 - q0]))

    return Si * Sj
