from __future__ import print_function, division
import numpy as np

import scipy.linalg as linalg

import figures
import generators.plane as plane

import matplotlib.pyplot as plt


def plane_tau0(t=1, parasite=0, phase=0, ylim1=None, ylim2=None, Es=None, directory="", yticks2=None):
    """generate g1 and g2 for plane"""
    Us = [0.5, 1, 2, 10]
    ses = [plane.UniformPlane(L=8, W=8, js=(56, 0), E=0, t=t, U=Ui * t, phase=phase, parasite=parasite) for Ui in Us]

    for i in range(0, len(Us)):
        ses[i].label = r'${}$'.format(ses[i].info['Us'][0] * t)

    if Es is None:
        Es = np.linspace(-8 / 2 * t, 8 / 2 * t, 512)

    extra = None
    if parasite > 0:
        extra = 'parasite-{}'.format(parasite)

    if phase > 0:
        extra += 'phase-{}'.format(phase)

    figures.g2_coherent(ses, (0, 0), (1, 1), Es, 2 * Es, ses[0].directory(extra) + directory, offset=0, ylims=[ylim1, ylim2], yticks2=yticks2)


def eigenbasis(se, nb):
    # generate number sector
    ns1 = se.model.numbersector(nb)

    # get the size of the basis
    ns1size = ns1.basis.len  # length of the number sector basis
    # G1i = range(ns1size)    # our Greens function?

    # self energy
    # sigma = self.sigma(nb, phi)

    # Effective Hamiltonian
    H1n = ns1.hamiltonian

    # Complete diagonalization
    E1, psi1r = linalg.eig(H1n.toarray(), left=False)
    psi1l = np.conj(np.linalg.inv(psi1r)).T
    # psi1l = np.conj(psi1r).T

    # check for dark states (throw a warning if one shows up)
    # if (nb > 0):
    #     Setup.check_for_dark_states(nb, E1)

    return E1, psi1l, psi1r


def edge_phase():
    """calculate edge phase"""
    se = plane.UniformPlane(L=8, W=8, js=(0, 8 * 7), E=0, t=1, U=0, phase=.2 * 2 * np.pi, parasite=.1)

    E1, psi1l, psi1r = eigenbasis(se, 1)
    idx = np.argsort(np.real(E1))
    E1 = E1[idx]
    psi1l = psi1l[:, idx]
    psi1r = psi1r[:, idx]

    res = np.zeros((64, ))
    idxs = se.edge_indices(dw=1, dl=1)
    print(idxs)
    s = len(idxs)

    for i in range(s):
        res += np.array([np.arctan2(np.real(psi1r[idxs[i], j] / psi1r[idxs[(i + 1) % s], j]), np.imag(psi1r[idxs[i], j] / psi1l[idxs[(i + 1) % s], j])) for j in np.arange(64)])

    plt.plot(np.real(E1), res / (2 * np.pi), '-o')
    Emin = np.min(np.real(E1))
    Emax = np.max(np.real(E1))
    for i in range(-10, 1, 1):
        plt.plot([Emin, Emax], [i, i])
        plt.plot([Emin, Emax], [-i, -i])
    plt.show()


def angle(z1, z2):
    """helper function"""
    return np.angle(z2 / z1)


def plot_eigen_function(ax, se, n, psi1l, psi1r):

    # plt.figure(figsize=(8, 8))
    for x in range(se.info['L']):
        for y in range(se.info['W']):
            i = x + y * se.info['L']

            w = np.sqrt(10) * np.abs(psi1r[i, n])
            arg = np.angle(psi1r[i, n])

            circle = plt.Circle((x, y), w, color='black', zorder=10)
            ax.add_artist(circle)

            ax.plot([x, x + w * np.cos(arg)], [y, y + w * np.sin(arg)], color='white', lw=.8, zorder=12)

        ax.set_xlim([-.5, se.info['L'] - .5])
        ax.set_ylim([-.5, se.info['W'] - .5])
    # plt.show()


def plane_eigenbasis(t=1, parasite=0, phase=0, n=0, Es=None, directory=""):
    """analyze"""
    se = plane.UniformPlane(L=8, W=8, js=(0, 8 * 7), E=0, t=t, U=0, phase=phase, parasite=parasite)
    print(se.model.links)
    # E1, psi1l, psi1r = se.eigenbasis(1)

    E1, psi1l, psi1r = eigenbasis(se, 1)

    # sort
    # print(E1)
    idx = np.argsort(np.real(E1))
    E1 = E1[idx]
    psi1l = psi1l[:, idx]
    psi1r = psi1r[:, idx]

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.plot(np.real(E1), np.imag(E1), 'o')
    ax1.plot(np.real(E1[n]), np.imag(E1[n]), 'o', color='red')

    ax2.plot(np.real(E1), [np.angle(psi1l[0, i] / psi1l[1, i]) for i in np.arange(64)], 'o')
    ax2.plot(np.real(E1[n]), np.angle(psi1l[0, n] / psi1l[1, n]), 'o', color='red')

    plt.show()

    def plot_weights(ax, se, n, psi1l, psi1r):
        for x in range(se.info['L']):
            for y in range(se.info['W']):
                i = x + y * se.info['L']
                w = psi1r[i, n].conjugate() * psi1l[i, n]
                circle = plt.Circle((x, y), np.abs(w) * 10, color='black', zorder=10)
                ax.add_artist(circle)
                # ax.text(x, y, '{0}'.format(x + y * se.info['L']), zorder=20, ha='center', va='center', color="#e5e5e5")

        ax.set_xlim([-.5, se.info['L'] - .5])
        ax.set_ylim([-.5, se.info['W'] - .5])

    fig = plt.figure(figsize=(8, 8))
    for n in range(se.info['L']):
        for m in range(se.info['W']):
            l = n + m * se.info['L']
            ax = plt.subplot(se.info['L'], se.info['W'], l + 1)
            plot_eigen_function(ax, se, l, psi1l, psi1r)

    plt.show()


def plane_tau(t=1, U=1, parasite=0, phase=0, taus=None, zticks=None, logscale=False):
    """tau dependence"""
    se = plane.UniformPlane(L=8, W=8, js=(0, 7), E=0, t=t, U=U, phase=phase, parasite=parasite)
    se.label = r'$U={0}\Gamma$'.format(U)

    omegas = np.linspace(-8 * t, 8 * t, 1024)

    if taus is None:
        taus = np.linspace(0, 4, 1024)

    if zticks is None:
        zticks = [0, 0.5, 1, 1.5, 2]

    extra = None
    if parasite > 0:
        extra = 'parasite-{}'.format(parasite)

    if phase > 0:
        extra += 'phase-{}'.format(phase)

    figures.g2_coherent_tau(se, (0, 0), (1, 1), omegas, taus, directory=se.directory(extra), ticks=zticks, logscale=logscale, title="for $U = {}\Gamma$".format(U), verbose=True)


def plane_g1(t=1, parasite=0, phase=0, ylim=None, Es=None, directory=""):
    """g1 function"""
    phis = [phase]

    ses = [plane.UniformPlane(L=8, W=8, js=(0, 7), E=0, t=t, U=0, phase=phis[0], parasite=parasite)]

    for i, phi in enumerate(phis):
        ses[i].label = r'$\phi={0}$'.format(phi)

    # se.label = r'${}$'.format(U)

    if Es is None:
        Es = np.linspace(-5 * t, 5 * t, 512)

    figures.g1_coherent(ses, 0, 1, Es, ses[0].directory() + directory, offset=0, ylim=ylim)


if __name__ == "__main__":
    print('-----')

    # edge_phase()
    # plane_eigenbasis(t=1, parasite=0.1, phase=.2 * 2 * np.pi, n=0)

    # t = Gamma
    plane_tau0(t=1, parasite=.1, phase=.2 * 2 * np.pi, ylim1=(1e-3, 2), ylim2=(1e-4, 1e4))
    # plane_eigenbasis(t=1, parasite=.3, phase=1 / 10 * 2 * np.pi, n=6)

    # plane_g1(t=1, parasite=.1, phase=.15 * np.pi * 2, ylim=(1e-6, 2))
    # plane_tau0(t=1, ylim1=(1e-5, 2), ylim2=(1e0, 1e30), Es = np.linspace(2 / 2 * 1, 12 / 2 * 1, 1024), directory='2-', yticks2=[1e0, 1e15, 1e30])

    # plane_tau(t=1, U=1, parasite=.1, phase=.2 * 2 * np.pi, zticks=[0, .5, 1, 1.5, 2], logscale=False, taus=np.linspace(0, 16, 1024))
    # plane_tau(t=1, U=10, parasite=.1, phase=.2 * 2 * np.pi, zticks=[1e-4, 1e-2, 1, 1e2, 1e4], logscale=True, taus=np.linspace(0, 16, 1024))

    # t = 10 Gamma
    # plane_tau0(t=10, ylim1=(1e-8, 2), ylim2=(1e-6, 1e14))
    # plane_tau0(t=10, ylim1=(1e-5, 2), ylim2=(1e0, 1e30), Es = np.linspace(2 / 2 * 10, 12 / 2 * 10, 1024), directory='2-', yticks2=[1e0, 1e15, 1e30])

    # plane_tau(t=10, U=1, zticks=[1e-2, 1e-1, 1, 1e1, 1e2], logscale=True, taus=np.linspace(0, 1.6, 1024))
