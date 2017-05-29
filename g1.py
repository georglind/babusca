from __future__ import division, print_function
import numpy as np
import smatrix


def g1(se, chli, chlo, Es):
    """
    Plots g1-correlation function.

    Parameters
    ----------
    se : scattering.Setup object
        The scattering setup
    chli : int
        Incoming channel
    chlo : int
        Outgoing channel
    Es : array-like
        List of single-photon energies
    """
    return np.abs(smatrix.one_particle(se, chli, chlo, Es)[1]) ** 2
