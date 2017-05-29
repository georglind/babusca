from __future__ import division, print_function
import numpy as np
import sys
import os
sys.path.append('/Users/kim/Science/Projects/Bose Hubbard Scattering/Numerics/')
import babusca.scattering as scattering


class QL(scattering.Setup):

    def __init__(self, N, omegas, Us, xs, gs=None, parasite=None):

        model = scattering.Model(
            Es=omegas,
            links=[],
            Us=Us)

        if gs is None:
            gs = normalize([1, 1])

        sites = np.arange(0, N)

        channels = []
        channels.append(scattering.Channel(sites=sites, strengths=gs, positions=xs))
        channels.append(scattering.Channel(sites=sites, strengths=gs, positions=-np.array(xs)))

        parasites = None
        if parasite is not None:
            parasites = [scattering.Channel(site=i, strength=parasite) for i in xrange(N)]

        scattering.Setup.__init__(self, model, channels, parasites)


def normalize(gs):
    gs = [g * 2 / (np.sqrt(np.pi) * (gs[0] + gs[1])) for g in gs]
    return gs
