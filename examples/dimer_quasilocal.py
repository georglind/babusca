from __future__ import print_function, division
import numpy as np
import os

import figures
import generators.proximate as prox


def weak_hybridization(phi=0, chlo=1, ylim1=None, ylim2=None):
    n = 2
    t = 1

    if ylim1 is None:
        ylims = None
    else:
        ylims = (ylim1, ylim2)

    offset = 1e3
    xs = np.array([0, phi / offset])

    Us = [.5, 1, 2, 10]
    ses = [prox.UniformProxChain(n, offset, t, Ui, xs=xs) for Ui in Us]

    for i in range(len(Us)):
        ses[i].label = r'${0}$'.format(ses[i].info['Us'][0])

    directory = ses[0].directory() + 'phi{}/'.format(phi)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    d1s = np.linspace(-5, 5, 512)
    d2s = np.linspace(-10, 10, 512)
    figures.g2_coherent(ses, (0, 0), (chlo, chlo), d1s=d1s, d2s=d2s, offset=offset, directory=directory, ylims=ylims)
    # figures.phi2_coherent(ses, (0, 0), (chlo, chlo), d1s=d1s, d2s=d2s, offset=offset, directory=directory)


def strong_hybridization(phi=0, ylim1=None, ylim2=None):
    if ylim1 is None:
        ylims = None
    else:
        ylims = (ylim1, ylim2)

    n = 2
    t = 10

    offset = 1e3
    xs = np.array([0, phi / offset])

    Us = [.5, 1, 4, 10]
    ses = [prox.UniformProxChain(n, offset, t, Ui, xs=xs) for Ui in Us]

    for i in range(len(Us)):
        ses[i].label = r'${0}$'.format(ses[i].info['Us'][0])

    directory = ses[0].directory() + 'phi{}/'.format(phi)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    d1s = np.linspace(-25, 25, 512)
    d2s = np.linspace(-50, 50, 512)
    figures.g2_coherent(ses, (0, 0), (1, 1), d1s=d1s, d2s=d2s, offset=offset, directory=directory, ylims=ylims)
    # figures.phi2_coherent(ses, (0, 0), (1, 1), d1s=d1s, d2s=d2s, offset=offset, directory=directory)


def W_weak_hybridization(phi=0, chlo=1, ylim1=None, ylim2=None):
    if ylim1 is None:
        ylims = None
    else:
        ylims = (ylim1, ylim2)

    n = 2
    t = 1

    offset = 1e3
    xs = np.array([0, phi / offset])

    Ws = [.5, 1, 4, 10]
    ses = [prox.UniformProxChain(n, offset, t, 1e8, Ws=np.full((n, n), Wi) - np.diag([Wi] * n), xs=xs) for Wi in Ws]

    for i in range(len(Ws)):
        ses[i].label = r'${0}$'.format(Ws[i])
        ses[i].info['name'] = 'W_' + ses[i].info['name']
        ses[i].info['Us'][0] = Ws[i]

    directory = ses[0].directory() + 'phi{}/'.format(phi)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    d1s = np.linspace(-8, 8, 512)
    d2s = np.linspace(-16, 16, 512)
    figures.g2_coherent(ses, (0, 0), (chlo, chlo), d1s=d1s, d2s=d2s, offset=offset, directory=directory, ylims=ylims)
    # figures.phi2_coherent(ses, (0, 0), (chlo, chlo), d1s=d1s, d2s=d2s, offset=offset, directory=directory)


def W_strong_hybridization(phi=0, chlo=1, ylim1=None, ylim2=None):
    if ylim1 is None:
        ylims = None
    else:
        ylims = (ylim1, ylim2)

    n = 2
    t = 10

    offset = 1e3
    xs = np.array([0, phi / offset])

    Ws = [.5, 1, 4, 10]
    ses = [prox.UniformProxChain(n, offset, t, 1e8, Ws=np.full((n, n), Wi) - np.diag([Wi] * n), xs=xs) for Wi in Ws]

    for i in range(len(Ws)):
        ses[i].label = r'${0}$'.format(Ws[i])
        ses[i].info['name'] = 'W_' + ses[i].info['name']
        ses[i].info['Us'][0] = Ws[i]

    directory = ses[0].directory() + 'phi{}/'.format(phi)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    d1s = np.linspace(-25, 25, 512)
    d2s = np.linspace(-50, 50, 512)

    figures.g2_coherent(ses, (0, 0), (chlo, chlo), d1s=d1s, d2s=d2s, offset=offset, directory=directory, ylims=ylims)
    # figures.phi2_coherent(ses, (0, 0), (chlo, chlo), d1s=d1s, d2s=d2s, offset=offset, directory=directory)


if __name__ == '__main__':

    if False:
        print('a')
        # weak_hybridization(.1 * np.pi, ylim1=(1e-3, 2), ylim2=(1e-3, 1e3))
    else:
        # weak_hybridization()
        # weak_hybridization(np.pi / 2 - .1 * np.pi)
        # weak_hybridization(np.pi)
        weak_hybridization(.1 * np.pi, 1, [1e-3, 2], [1e-3, 1e3])
        # weak_hybridization(np.pi / 2)
        # weak_hybridization(np.pi / 50)
        # weak_hybridization(np.pi / 100)

        # strong_hybridization()
        # strong_hybridization(np.pi)
        # strong_hybridization(np.pi / 2)
        # strong_hybridization(np.pi / 10)
        # strong_hybridization(np.pi / 100)
        # strong_hybridization(.4 * np.pi)

        # W_weak_hybridization()
        # W_weak_hybridization(np.pi / 2 - .1 * np.pi)
        # W_weak_hybridization(np.pi)
        # W_weak_hybridization(.1 * np.pi)
        # W_weak_hybridization(np.pi / 2)
        # W_weak_hybridization(np.pi / 50)
        # W_weak_hybridization(np.pi / 100)

        # W_strong_hybridization()
        # W_strong_hybridization(np.pi / 2 - .1 * np.pi)
        # W_strong_hybridization(np.pi)
        # W_strong_hybridization(.1 * np.pi)
        # W_strong_hybridization(np.pi / 2)
        # W_strong_hybridization(np.pi / 50)
        # W_strong_hybridization(np.pi / 100)
