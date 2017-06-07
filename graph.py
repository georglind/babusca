from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, nodes, xy=None, links=None):
        self.N = len(nodes)
        self.nodes = nodes

        if xy is None:
            xy = np.zeros((self.N, 2))
        self.xy = xy

        if links is None:
            links = []

        self.links = links

        neighbors = []
        for i in xrange(self.N):
            n = [j[0] for j in links if j[1] == i] + [j[1] for j in links if j[0] == i]
            neighbors.append(n)
        self.neighbors = neighbors

    def forcedirectedlayout(self):
        converged, self.xy = FD(self)

    def plot(self, fig=None, ax=None):
        xy = self.xy[:, :]

        xymax = np.max(xy, axis=0)
        xymin = np.min(xy, axis=0)

        # imin = [np.where(xy[:, i] == xymin[i])[0] for i in xrange[2]]
        xy -= np.reshape(np.tile(.5 * (xymax + xymin), self.N), (self.N, 2))

        xyscale = np.max((xymax - xymin) / 100)

        show = False
        if fig is None or ax is None:
            show = True
            fig, ax = plt.subplots(figsize=(4, 4))

        for link in self.links:
            ax.plot(xy[link[0:2], 0] / xyscale, xy[link[0:2], 1] / xyscale, 'b')

        for i in xrange(self.N):
            x, y = xy[i, 0:2] / xyscale
            circle = plt.Circle((x, y), 3, color='#e5e5e5', zorder=10)
            ax.add_artist(circle)
            ax.text(x, y + 1, '{0}'.format(i), zorder=20, ha='center', va='center')

        ax.set_xlim([-60, 60])
        ax.set_ylim([-60, 60])

        if show is True:
            plt.show()


def FD(graph):
    energy = 0
    heat = 0.02

    xy = 1 * np.random.rand(graph.N, 2)
    vs = np.zeros((graph.N, 2))

    settings = {'vmax': 1, 'friction': .2}

    cnt = 0
    converged = False
    while not converged:
        cnt += 1

        if heat > 0:
            if cnt == 20:
                heat = .01
            elif cnt == 40:
                heat = .005
            elif cnt == 80:
                heat = 0

        xy, vs, energy = FDstep(graph, xy, vs, energy, heat, settings)

        if energy > 100 * graph.N:
            break

        # print(np.sum(np.abs(vs))/graph.N)
        converged = np.sum(np.abs(vs)) / graph.N < 1e-3
        if cnt > 500:
            converged = True

    print(cnt)
    return converged, xy


def FDstep(graph, xy=None, vs=None, energy=0, heat=0.5, settings=None):
    N = graph.N

    if xy is None:
        xy = np.random.rand(N, 2)
    if vs is None:
        vs = np.zeros((N, 2))

    # xy += -np.reshape(np.tile(np.sum(xy, axis=0)/N, N), (N, 2))

    nenergy = 0
    rforce = np.zeros((N, 2))  # repelling force
    aforce = np.zeros((N, 2))  # attractive force

    for i in xrange(N):
        for j in xrange(N):
            if i == j:
                continue
            d = xy[i, :] - xy[j, :]
            r = np.sqrt(np.sum(d ** 2))
            rforce[i, :] += 0.001 * d / r ** 3 if r > 0.001 else .1 * (np.random.rand(2) - .5)
            nenergy += 0.005 / max(r, 0.001)  # new energy

    for i in xrange(N):
        for j in graph.neighbors[i]:
            d = xy[i, :] - xy[j, :]
            r = np.sqrt(np.sum(d ** 2))
            aforce[i, :] += -2 * d / 100 if r > 0.001 else .1 * (np.random.rand(2) - .5)
            nenergy += r ** 2 / 100

    vs += (aforce + rforce)

    # idx = np.where(np.abs(vs) > settings['vmax'])
    # vs[idx] = settings['vmax']*np.sign(vs[idx])

    vs -= settings['friction'] * vs
    # idx = np.where(np.abs(vs) > settings['friction'])
    # vs[idx] += -settings['friction']*np.sign(vs[idx])

    if heat > 0:
        vs += 2 * heat * (2 * np.random.rand(N, 2) - 1)

    xy += vs
    # if heat > 0:
    #     xy +=

    xy += -np.reshape(np.tile(np.sum(xy, axis=0), N) / N, (N, 2))

    # print(vs)
    # graph.xy = xy
    # graph.plot()

    return xy, vs, nenergy


if __name__ == "__main__":
    # This is a demonstrating of the graph library showing a plot of a 7 site chain using force-directed layout.

    nodes = [0] * 7
    links = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [4, 6], [6, 5]]

    g = Graph(nodes, None, links)
    g.forcedirectedlayout()
    g.plot()
