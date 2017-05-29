from __future__ import print_function, division
import numpy as np

import figures
import generators.cube as cube


def cube_tau0(t=1, ylim1=None, ylim2=None, Es=None, directory="", yticks2=None):
    Us = [0.5, 4, 20]
    ses = [cube.UniformCube(L=4, W=4, H=4, js=(0, 3), E=0, t=t, U=Ui * t) for Ui in Us]

    for i in xrange(0, len(Us)):
        ses[i].label = r'${}$'.format(ses[i].info['Us'][0] * t)

    if Es is None:
        Es = np.linspace(-5 / 2 * t, 25 / 2 * t, 1024)

    figures.g2_coherent(ses, (0, 0), (1, 1), Es, 2 * Es, ses[0].directory() + directory, offset=0, ylims=[ylim1, ylim2], yticks2=yticks2)


def cube_tau(t=1, U=1, taus=None, zticks=None, logscale=False):
    se = cube.UniformCube(L=4, W=4, H=4, js=(0, 3), E=0, t=t, U=U)
    se.label = r'$U={0}\Gamma$'.format(U)

    omegas = np.linspace(-5 * t, 15 * t, 1024)

    if taus is None:
        taus = np.linspace(0, 4, 1024)

    if zticks is None:
        zticks = [0.5, 0.75, 1, 1.25, 1.5]

    figures.g2_coherent_tau(se, (0, 0), (1, 1), omegas, taus, directory=se.directory(), ticks=zticks, logscale=logscale, title="for $U = {}\Gamma$".format(U))


if __name__ == "__main__":
    # t = Gamma
    cube_tau0(t=1, ylim1=(1e-6, 2), ylim2=(1e-5, 1e12))
    # cube_tau0(t=1, ylim1=(1e-5, 2), ylim2=(1e0, 1e30), Es = np.linspace(2 / 2 * 1, 12 / 2 * 1, 1024), directory='2-', yticks2=[1e0, 1e15, 1e30])

    # cube_tau(t=1, U=1, zticks=[1e-2, 1e-1, 1, 1e1, 1e2], logscale=True, taus=np.linspace(0, 16, 1024))

    # t = 10 Gamma
    # cube_tau0(t=10, ylim1=(1e-8, 2), ylim2=(1e-6, 1e14))
    # cube_tau0(t=10, ylim1=(1e-5, 2), ylim2=(1e0, 1e30), Es = np.linspace(2 / 2 * 10, 12 / 2 * 10, 1024), directory='2-', yticks2=[1e0, 1e15, 1e30])

    # cube_tau(t=10, U=1, zticks=[1e-2, 1e-1, 1, 1e1, 1e2], logscale=True, taus=np.linspace(0, 1.6, 1024))
