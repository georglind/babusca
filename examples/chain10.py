from __future__ import print_function, division
import sys
import numpy as np
sys.path.append('../../')

import figures
import babusca.generators.chain as chain


def chain_tau0(t=1, ylim1=None, ylim2=None, Es=None, directory="", yticks2=None, parasite=0):
    Us = [0.5, 1, 2, 10]
    ses = [chain.UniformChain(N=10, js=(0, 9), E=0, t=t, U=Ui * t, parasite=parasite) for Ui in Us]

    for i in range(0, len(Us)):
        ses[i].label = r'${}$'.format(ses[i].info['Us'][0])

    if Es is None:
        Es = np.linspace(-5 / 2 * t, 15 / 2 * t, 1024)

    figures.g2_coherent(ses, (0, 0), (1, 1), Es, 2 * Es, ses[0].directory() + directory, offset=0, ylims=[ylim1, ylim2], yticks2=yticks2)


def chain_tau(t=1, U=1, parasite=0, taus=None, zticks=None, logscale=False):
    se = chain.UniformChain(N=10, js=(0, 9), E=0, t=t, U=U, parasite=parasite)
    se.label = r'$U={0}\Gamma$'.format(U)

    omegas = np.linspace(-5 * t, 5 * t, 1024)

    if taus is None:
        taus = np.linspace(0, 4, 1024)

    if zticks is None:
        zticks = [0.5, 0.75, 1, 1.25, 1.5]

    figures.g2_coherent_tau(se, (0, 0), (1, 1), omegas, taus, directory=se.directory(), ticks=zticks, title="for $U = {}\Gamma$".format(U), logscale=logscale)


if __name__ == "__main__":
    # t = Gamma
    chain_tau0(t=1, ylim1=(1e-5, 2), ylim2=(1e-5, 1e5), parasite=0)
    chain_tau0(t=1, ylim1=(1e-5, 2), ylim2=(1e0, 1e30), Es=np.linspace(2 / 2 * 1, 12 / 2 * 1, 1024), directory='2-', yticks2=[1e0, 1e15, 1e30])

    # chain_tau(t=1, U=1, parasite=0, zticks=[0, .5, 1, 1.5, 2], taus=np.linspace(0, 6, 1024), logscale=False)

    # t = 10 Gamma
    chain_tau0(t=10, ylim1=(1e-5, 2), ylim2=(1e-5, 1e5), parasite=.5)
    chain_tau0(t=10, ylim1=(1e-5, 2), ylim2=(1e0, 1e30), Es=np.linspace(2 / 2 * 10, 12 / 2 * 10, 1024), directory='2-', yticks2=[1e0, 1e15, 1e30], parasite=.5)

    # chain_tau(t=10, U=10, parasite=.5, zticks=[0, .5, 1, 1.5, 2], taus=np.linspace(0, 3, 1024), logscale=False)
