from __future__ import print_function, division
import sys
import numpy as np

import context
import figures
import generators.chain as chain


def dimer_tau0(js, t=1, ylim1=None, ylim2=None):
    Us = [0.5, 1, 2, 10]
    ses = [chain.UniformChain(N=2, js=js, E=0, t=t, U=Ui * t) for Ui in Us]

    for i in range(0, len(Us)):
        ses[i].label = r'${}$'.format(ses[i].info['Us'][0])

    Es = np.linspace(-5 / 2 * t, 15 / 2 * t, 1024)

    figures.g2_coherent(ses, (0, 0), (1, 1), Es, 2 * Es, ses[0].directory(), offset=0, ylims=[ylim1, ylim2])


def dimer_tau(js, t=1, U=1, taus=None):
    se = chain.UniformChain(N=2, js=js, E=0, t=t, U=U)
    se.label = r'$U={0}\Gamma$'.format(U)

    omegas = np.linspace(-5 * t, 15 * t, 1024)

    if taus is None:
        taus = np.linspace(0, 4, 1024)

    figures.g2_coherent_tau(se, (0, 0), (1, 1), omegas, taus, directory=se.directory(), ticks=[0.5, 0.75, 1, 1.25, 1.5], title="for $U = {}\Gamma$".format(U))


if __name__ == '__main__':

    # strongly coupled t = Gamma
    dimer_tau0((0, 1), 1, (1e-2, 2), (1e-3, 1e3))
    # dimer_tau((0, 1), 1, 1)

    # weakly coupled t = 10 Gamma
    dimer_tau0((0, 1), 10, (1e-5, 2), (1e-4, 1e6))
    # dimer_tau((0, 1), 10, 1, taus=np.linspace(0, 1, 1024))

    # same site
    dimer_tau0((0, 0), 1, (1e-3, 2), (1e-1, 1e4))
