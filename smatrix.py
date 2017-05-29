from __future__ import division, print_function
import numpy as np
import scattering
import tmatrix

#      $$$$$$\                                  $$\               $$\
#     $$  __$$\                                 $$ |              \__|
#     $$ /  \__|       $$$$$$\$$$$\   $$$$$$\ $$$$$$\    $$$$$$\  $$\ $$\   $$\
#     \$$$$$$\ $$$$$$\ $$  _$$  _$$\  \____$$\\_$$  _|  $$  __$$\ $$ |\$$\ $$  |
#      \____$$\\______|$$ / $$ / $$ | $$$$$$$ | $$ |    $$ |  \__|$$ | \$$$$  /
#     $$\   $$ |       $$ | $$ | $$ |$$  __$$ | $$ |$$\ $$ |      $$ | $$  $$<
#     \$$$$$$  |       $$ | $$ | $$ |\$$$$$$$ | \$$$$  |$$ |      $$ |$$  /\$$\
#      \______/        \__| \__| \__| \_______|  \____/ \__|      \__|\__/  \__|


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
    """
    Single particle

    Parameters
    ----------
    s : Setup
        Scattering setup
    chi : int
        Input channel
    cho : int
        Output channel
    Es : ndarray
        Input energies
    """
    Es, T1 = tmatrix.one_particle(sp, chli, chlo, Es)

    S1 = np.zeros(T1.shape, dtype=np.complex128)

    if chli == chlo:
        S1 += 1

    S1 -= 2 * 1j * np.pi * T1

    return Es, S1


def twp_photon(sp, chlsi, chlso, E, dE, qs=None):
    """alias for two_particle"""
    return two_particle(s, chlsi, chlso, E, dE, qs)


def two_particle(s, chlsi, chlso, E, dE, qs=None):
    """
    Calculates the 2-particle scattering matrix S(2)

    Parameters
    ----------
    m : bosehubbard.Model object
        Describes the Bose-Hubbard Model
    chanis : list
        Input state channels (in) [k0, k1]
    chanos : list
        Output state channels (out) [k'0, k'1]
    E : float
        Total two-particle energy
    dE : float
        Energy difference between the two input particles
    """
    # relevant energy range
    if qs is None:
        qs = scattering.energies(E, dE)

    # single particle scattering matrix
    S1 = np.zeros((2, 2, 2), dtype=np.complex128)
    for i, chi in enumerate(chlsi):
        for j, cho in enumerate(chlso):
            _, S1[i, j, :] = one_particle(s,
                                          chi,
                                          cho,
                                          Es=.5 * np.array([E - dE, E + dE]))

    # S(1) * S(1)
    D2 = np.array([S1[0, 1, 0] * S1[1, 0, 1], S1[0, 0, 0] * S1[1, 1, 1]])

    # T2 matrix
    _, T2, T2in = tmatrix.two_particle(s, chlsi, chlso, E, dE, qs)

    # Factor of 4 due to symmetries
    S2 = - 4 * 2 * np.pi * 1j * T2

    return qs, D2, S2, -2 * np.pi * 1j * T2in
