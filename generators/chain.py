from __future__ import division, print_function
import numpy as np
import sys
import os

# babusca scattering
from context import scattering


class Chain(scattering.Setup):

    def __init__(self, N, js, Es, ts, Us, gs=None, parasite=0):

        model = scattering.Model(
            Es=Es,
            links=[(i, i + 1, ts[i]) for i in xrange(N - 1)],
            Us=Us)

        if gs is None:
            gs = normalize([1, 1])

        channels = []
        channels.append(scattering.Channel(site=js[0], strength=gs[0]))
        channels.append(scattering.Channel(site=js[1], strength=gs[1]))

        parasites = []
        if parasite > 0:
            parasites = [scattering.Channel(site=i, strength=parasite) for i in xrange(N)]

        scattering.Setup.__init__(self, model, channels, parasites)

        self.info = {
            'N': N,
            'js': js,
            'Es': Es,
            'ts': ts if len(ts) > 0 else [0],
            'Us': Us,
            'gs': gs,
            'parasite': parasite,
            'name': 'chain'
        }

    def directory(self, extra=None):
        dd = self.info['name'] + '{0}/{1}/t{2}/'.format(self.info['N'], self.info['js'], self.info['ts'][0])
        dd += 'U{0}/'.format(self.info['Us'][0])

        if extra is not None:
            dd += '{}/'.format(extra)

        if not os.path.isdir(dd):
            os.makedirs(dd)

        return dd


class ImpurityChain(Chain):

    def __init__(self, N, js, Es, t, U, gs=None, parasite=0):
        Es = [Es[0]] + [0] * (N - 2) + [Es[1]]
        Us = [U] + [0] * (N - 2) + [U]
        Chain.__init__(self, N, js, Es, [t] * (N - 1), Us, gs, parasite)
        self.info['name'] = 'impuritychain'


class UniformChain(Chain):

    def __init__(self, N, js, E, t, U, gs=None, parasite=0):
        Chain.__init__(self, N, js, [E] * N, [t] * (N - 1), [U] * N, gs, parasite)
        self.info['name'] = 'uniformchain'


def noisify(chse, dE, dt, dU):
    """
    Create a nosified chain

    Parameters
    chse : Chain instance
    cE: float
        spread in on-site energies
    dt: float
        spread in hoppings
    dU: float
        spread in interaction
    """
    dEs = np.zeros((chse.N,))
    dts = np.zeros((chse.N,))
    dUs = np.zeros((chse.N,))

    if dE > 0:
        dEs = np.random.uniform(-dE, dE, size=(chse.N,))
        dEs = dEs - np.average(dEs)
    if dt > 0:
        dts = np.random.normal(scale=dt, size=(chse.N - 1,))
    if dU > 0:
        dUs = np.random.normal(scale=dU, size=(chse.N,))

    # print(dEs)

    c = Chain(chse.N, chse.js,
              [chse.Es[i] + dEs[i] for i in xrange(chse.N)],
              [chse.ts[i] + dts[i] for i in xrange(chse.N - 1)],
              [chse.Us[i] + dUs[i] for i in xrange(chse.N)],
              chse.gs, chse.parasites[0].strength)

    c.info['name'] = 'noisychain'
    return c


def normalize(gs):
    gs = [g * 2 / (np.sqrt(np.pi) * (gs[0] + gs[1])) for g in gs]
    return [gs[0] + 0.0001, gs[1] + 0.0001]
