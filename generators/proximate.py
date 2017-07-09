from __future__ import division, print_function
import numpy as np
import os

# import babusca scattering
from context import scattering


# quasi-locally coupled chain
class ProximateChain(scattering.Setup):

    def __init__(self, N, Es, ts, Us, Ws=None, xs=None, gs=None, parasite=0):

        # defaults
        Ws = np.zeros((N, N)) if Ws is None else Ws
        xs = np.zeros((N, 1)) if xs is None else xs

        # model
        model = scattering.Model(
            Es=Es,
            links=[(i, i + 1, ts[i]) for i in range(N - 1)],
            Us=Us,
            W=Ws)

        # coupling strengths
        if gs is None:
            gs = normalize([1, 1])

        # sites
        sites = np.arange(0, N)

        channels = []
        channels.append(scattering.Channel(sites=sites, strengths=gs, positions=xs))
        channels.append(scattering.Channel(sites=sites, strengths=gs, positions=-np.array(xs)))

        parasites = None
        if parasite > 0:
            parasites = [scattering.Channel(site=i, strength=parasite) for i in range(N)]

        scattering.Setup.__init__(self, model, channels, parasites)

        self.info = {
            'N': N,
            'Es': Es,
            'ts': ts,
            'Us': Us,
            'Ws': Ws,
            'gs': gs,
            'xs': xs,
            'parasite': parasite,
            'name': 'proximatechain'
        }

    def directory(self, extra=None):
        dd = self.info['name'] + '{0}/t{1}/'.format(self.info['N'], self.info['ts'][0])
        dd += 'U{0}/'.format(self.info['Us'][0])

        if extra is not None:
            dd += '{}/'.format(extra)

        if not os.path.isdir(dd):
            os.makedirs(dd)

        return dd


class ImpurityProxChain(ProximateChain):

    def __init__(self, N, Es, t, U, Ws=None, xs=None, gs=None, parasite=0):
        Es = [Es[0]] + [0] * (N - 2) + [Es[1]]
        Us = [U] + [0] * (N - 2) + [U]
        ProximateChain.__init__(self, N, Es, [t] * (N - 1), Us, Ws, xs, gs, parasite)
        self.info['name'] = 'impureproxchain'


class UniformProxChain(ProximateChain):

    def __init__(self, N, E, t, U, Ws=None, xs=None, gs=None, parasite=0):
        ProximateChain.__init__(self, N, [E] * N, [t] * (N - 1), [U] * N, Ws=Ws, xs=xs, gs=gs, parasite=parasite)
        self.info['name'] = 'uniproxchain'


def normalize(gs):
    gs = [g * 2 / (np.sqrt(np.pi) * (gs[0] + gs[1])) for g in gs]
    return [gs[0], gs[1]]
