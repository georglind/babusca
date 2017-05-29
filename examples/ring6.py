from __future__ import print_function, division
import os
import numpy as np

import figures
import generators.ring as ring


def ring_tau0(t=1, ylim1=None, ylim2=None, Es=None, directory="", yticks2=None):
    Us = [0.5, 1, 2, 10]
    ses = [ring.UniformRing(N=6, js=(0, 2), E=0, t=t, U=Ui * t) for Ui in Us]

    for i in xrange(0, len(Us)):
        ses[i].label = r'${}$'.format(ses[i].info['Us'][0] * t)

    if Es is None:
        Es = np.linspace(-5 / 2 * t, 15 / 2 * t, 1024)

    figures.g2_coherent(ses, (0, 0), (1, 1), Es, 2 * Es, ses[0].directory() + directory, offset=0, ylims=[ylim1, ylim2], yticks2=yticks2)


def ring_tau(t=1, U=1, taus=None, zticks=None):
    se = ring.UniformRing(N=6, js=(0, 2), E=0, t=t, U=U)
    se.label = r'$U={0}\Gamma$'.format(U)

    omegas = np.linspace(-5 * t, 15 * t, 1024)

    if taus is None:
        taus = np.linspace(0, 4, 1024)

    if zticks is None:
        zticks = [0.5, 0.75, 1, 1.25, 1.5]

    figures.g2_coherent_tau(se, (0, 0), (1, 1), omegas, taus, directory=se.directory(), ticks=zticks, title="for $U = {}\Gamma$".format(U))


if __name__ == "__main__":
    # t = Gamma
    ring_tau0(t=1, ylim1=(1e-4, 2), ylim2=(1e-3, 1e7))
    # ring_tau0(t=1, ylim1=(1e-5, 2), ylim2=(1e0, 1e30), Es = np.linspace(2 / 2 * 1, 12 / 2 * 1, 1024), directory='2-', yticks2=[1e0, 1e15, 1e30])

    # ring_tau(t=1, U=1, zticks=[0, .5, 1, 1.5, 2.0], taus=np.linspace(0, 10, 1024))

    # t = 10 Gamma
    ring_tau0(t=10, ylim1=(1e-6, 2), ylim2=(1e-5, 1e8))
    # ring_tau0(t=10, ylim1=(1e-5, 2), ylim2=(1e0, 1e30), Es = np.linspace(2 / 2 * 10, 12 / 2 * 10, 1024), directory='2-', yticks2=[1e0, 1e15, 1e30])

    # ring_tau(t=10, U=1, zticks=[0, .5, 1, 1.5, 2.0], taus=np.linspace(0, 10, 1024))
