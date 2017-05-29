from __future__ import division, print_function
import numpy as np

# import babusca scattering
from context import scattering


class Plane(scattering.Setup):

    def __init__(self, L, W, js, Es, ts, Us, gs=None, parasite=0):

        links = []
        i = 0
        for x in xrange(L):
            for y in xrange(W):
                n = x + L * y
                if x < L - 1:
                    links.append([n, n + 1, ts[i]])
                    i += 1
                if y < W - 1:
                    links.append([n, n + L, ts[i]])
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

        N = L * W

        parasites = []
        if parasite > 0:
            parasites = [scattering.Channel(site=i, strength=parasite) for i in xrange(N)]

        scattering.Setup.__init__(self, model, channels, parasites)

        self.info = {}
        self.info['N'] = W * L
        self.info['L'] = L  # length of chain
        self.info['W'] = W
        self.info['js'] = js
        self.info['Es'] = Es
        self.info['ts'] = ts
        self.info['Us'] = Us
        self.info['gs'] = gs
        self.info['parasite'] = parasite
        self.info['name'] = 'plane'

    def directory(self, extra=None):
        dd = self.info['name'] + '{0}x{1}/{2}/t{3}/'.format(self.info['L'], self.info['W'], self.info['js'], self.info['ts'][0])
        dd += 'U{0}/'.format(self.info['Us'][0])

        if extra is not None:
            dd += '{}/'.format(extra)

        if not os.path.isdir(dd):
            os.makedirs(dd)

        return dd

    def edge_indices(self, dl=0, dw=0):
        # width and length
        L, W = self.info['L'], self.info['W']

        # rescale
        Ln = L - 2 * dl
        Wn = W - 2 * dw
        Nn = Ln * Wn + 2 * (Ln - 1) * dw
        edges = range(Wn - 1) + range(Wn - 1, Nn - 1, W) + range(Nn - 1, Nn - Wn, -1) + range(Nn - Wn, 0, -W)
        edges = [edge + dl * W + dw for edge in edges]

        return edges


class UniformPlane(Plane):

    def __init__(self, L, W, js, E, t, U, phase=0, gs=None, parasite=0):
        N = L * W
        Nl = 2 * (L - 1) * (W - 1) + L - 1 + W - 1

        if phase == 0:
            ts = [t] * Nl
        else:
            ts = []
            i = 0
            for x in xrange(L):
                for y in xrange(W):
                    phiy = phase * x
                    n = x + L * y
                    if x < L - 1:
                        ts.append(t)
                    if y < W - 1:
                        ts.append(t * np.exp(1j * phiy))
                    i += 1

        Plane.__init__(self, L, W, js, [E] * N, ts, [U] * N, gs, parasite)
        self.info['name'] = 'uniformplane'


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

    # print(dEs)
    Nl = len(chse.ts)
    c = Plane(chse.L, chse.W, chse.js,
              [chse.Es[i] + dEs[i] for i in xrange(chse.N)],
              [chse.ts[i] + dts[i] for i in xrange(Nl)],
              [chse.Us[i] + dUs[i] for i in xrange(chse.N)],
              chse.gs, chse.parasites[0].strength)

    c.info['name'] = 'noisyplane'
    return c


def normalize(gs):
    gs = [g * 2 / (np.sqrt(np.pi) * (gs[0] + gs[1])) for g in gs]
    return [gs[0] + 0.0001, gs[1] + 0.0001]
