from __future__ import division, print_function
import numpy as np
import os

# import babusca scattering
from context import scattering

# ring
class Ring(scattering.Setup):

    def __init__(self, N, js, Es, ts, Us, Ws=None, gs=None, parasite=0):

        model = scattering.Model(
            Es=Es,
            links=[(i, (i + 1) % N, ts[i]) for i in range(N)],
            Us=Us,
            W=Ws)

        if gs is None:
            gs = normalize([1, 1])

        channels = []
        channels.append(scattering.Channel(site=js[0], strength=gs[0]))
        channels.append(scattering.Channel(site=js[1], strength=gs[1]))

        parasites = []
        if parasite > 0:
            parasites = [scattering.Channel(site=i, strength=parasite) for i in range(N)]

        scattering.Setup.__init__(self, model, channels, parasites)

        self.info = {
            'N': N,
            'js': js,
            'Es': Es,
            'ts': ts,
            'Us': Us,
            'gs': gs,
            'parasite': parasite,
            'name': 'ring'
        }

    def directory(self, extra=None):
        dd = self.info['name'] + '{0}/{1}/t{2}/'.format(self.info['N'], self.info['js'], self.info['ts'][0])
        dd += 'U{0}/'.format(self.info['Us'][0])

        if extra is not None:
            dd += '{}/'.format(extra)

        if not os.path.isdir(dd):
            os.makedirs(dd)

        return dd


class UniformRing(Ring):

    def __init__(self, N, js, E, t, U, gs=None, parasite=0):
        Ring.__init__(self, N, js, [E] * N, [t] * N, [U] * N, gs=gs, parasite=parasite)
        self.info['name'] = 'uniformring'


def noisify(ring, dE, dt, dU):
    """
    Create a nosified ring

    Parameters
    chse : Ring instance
    cE: float
        spread in on-site energies
    dt: float
        spread in hoppings
    dU: float
        spread in interaction
    """
    dEs = np.random.normal(scale=dE, size=(ring.N,))
    dts = np.random.normal(scale=dt, size=(ring.N,))
    dUs = np.random.normal(scale=dU, size=(ring.N,))

    c = Ring(ring.N, ring.js,
             [ring.Es[i] + dEs[i] for i in range(ring.N)],
             [ring.ts[i] + dts[i] for i in range(ring.N)],
             [ring.Us[i] + dUs[i] for i in range(ring.N)],
             ring.gs, ring.parasites[0].strength)

    c.info['name'] = 'noisyring'
    return c


def normalize(gs):
    gs = [g * 2 / (np.sqrt(np.pi) * (gs[0] + gs[1])) for g in gs]
    return gs
