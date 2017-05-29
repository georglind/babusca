from __future__ import division, print_function
import numpy as np
import os

# import babusca scattering
from context import scattering


class Cube(scattering.Setup):

    def __init__(self, L, W, H, js, Es, ts, Us, gs=None, parasite=0):

        N = W * L * H  # number of sites

        # construct links list
        links = []

        i = 0
        for x in xrange(L):
            for y in xrange(W):
                for z in xrange(H):
                    n = x + L * y + W * L * z
                    if x < L - 1:
                        links.append([n, n + 1, ts[i]])
                    if y < W - 1:
                        links.append([n, n + L, ts[i]])
                    if z < H - 1:
                        links.append([n, n + L * W, ts[i]])
                    i += 1

        model = scattering.Model(
            Es=Es,
            links=links,
            Us=Us)

        if gs is None:
            gs = normalize([1, 1])

        channels = []
        channels.append(scattering.Channel(site=js[0], strength=gs[0]))
        channels.append(scattering.Channel(site=js[1], strength=gs[1]))

        parasites = []
        if np.abs(parasite) > 0:
            parasites = [scattering.Channel(site=i, strength=parasite) for i in xrange(N)]

        scattering.Setup.__init__(self, model, channels, parasites)

        self.info = {
            'N': N,
            'L': L,
            'W': W,
            'H': H,
            'js': js,
            'Es': Es,
            'ts': ts,
            'Us': Us,
            'gs': gs,
            'parasite': parasite,
            'name': 'cube'
        }

    def directory(self, extra=None):
        dd = self.info['name'] + '{0}x{1}x{2}/{3}/t{4}/'.format(self.info['L'], self.info['W'], self.info['H'], self.info['js'], self.info['ts'][0])
        dd += 'U{0}/'.format(self.info['Us'][0])

        if extra is not None:
            dd += '{}/'.format(extra)

        if not os.path.isdir(dd):
            os.makedirs(dd)

        return dd


class UniformCube(Cube):

    def __init__(self, L, W, H, js, E, t, U, gs=None, parasite=0):
        N = L * W * H
        Nl = (L - 1) * W * H + L * (W - 1) * H + L * W * (H - 1)

        Cube.__init__(self, L, W, H, js, [E] * N, [t] * Nl, [U] * N, gs, parasite)
        self.info['name'] = 'uniformcube'


def noisify(chse, dE, dt, dU):
    dEs = np.zeros((chse.N,))
    dts = np.zeros((len(chse.ts),))
    dUs = np.zeros((chse.N,))

    if dE > 0:
        dEs = np.random.uniform(-dE, dE, size=(chse.N,))
        dEs = dEs - np.average(dEs)
    if dt > 0:
        dts = np.random.normal(scale=dt, size=(len(chse.ts),))
    if dU > 0:
        dUs = np.random.normal(scale=dU, size=(chse.N,))

    ps = 0
    if len(chse.parasites) > 0:
        ps = chse.parasites[0].strength

    # print(dEs)
    Nl = len(chse.ts)
    c = Cube(chse.L, chse.W, chse.H, chse.js,
             [chse.Es[i] + dEs[i] for i in xrange(chse.N)],
             [chse.ts[i] + dts[i] for i in xrange(Nl)],
             [chse.Us[i] + dUs[i] for i in xrange(chse.N)],
             chse.gs, ps)

    c.info['name'] = 'noisycube'
    return c


def normalize(gs):
    gs = [g * 2 / (np.sqrt(np.pi) * (gs[0] + gs[1])) for g in gs]
    return [gs[0] + 0.0001, gs[1] + 0.0001]
