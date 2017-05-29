from __future__ import division, print_function
import numpy as np
import sys
sys.path.append('/Users/kim/Science/Software/bosehubbard/bosehubbard')
# import bosehubbard
# import scattering as scat
# import utilities as util
# import scipy.linalg as linalg
# import scipy.sparse as sparse
import smatrix
import matplotlib.pyplot as plt



#                $$$$$$\
#               $$  __$$\
#      $$$$$$\  \__/  $$ |
#     $$  __$$\  $$$$$$  |
#     $$ /  $$ |$$  ____/
#     $$ |  $$ |$$ |
#     \$$$$$$$ |$$$$$$$$\
#      \____$$ |\________|
#     $$\   $$ |
#     \$$$$$$  |
#      \______/


def g2_coherent_s(s, chlsi, chlso, E, qs=None):
    """
    g2 for a coherent state (or partially coherent)

    Parameters
    ----------
    s : Setup
        Scattering setup
    chlsi : list
        List of incoming channel indices
    chlso : list
        List of outgoing channel indices
    E : float
        Energy of the two-particle state
    """
    qs, D2, S2, _ = smatrix.two_particle(s, chlsi=chlsi, chlso=chlso, E=E, dE=0, qs=qs)

    # Single particle scattering
    Ee = np.array([.5 * E])
    _, S10 = smatrix.one_particle(s, chlsi[0], chlso[0], Ee)
    _, S11 = smatrix.one_particle(s, chlsi[1], chlso[1], Ee)

    dFS2 = np.abs(S10[0]) ** 2 * np.abs(S11[0]) ** 2

    # Fourier transform
    FS2 = np.fft.fft(S2) * (qs[1] - qs[0])

    # times
    times = np.fft.fftfreq(len(qs), d=qs[1] - qs[0])

    # add delta function contribution
    FS2 += np.exp(2 * np.pi * 1j * E / 2 * times) * np.sum(D2)

    return np.abs(FS2[0]) ** 2, dFS2


def g2(s, chlsi, chlso, E, dE):
    """
    Calculate the intensity-intensity correlation function as a function of total energy,
    E, and energy discrepancy, dE.

    Parameters
    ----------
    s : Setup
        Scattering setup object.
    chlsi : list
        Input state channel indices.
    chlso : list
        Output state channel indices.
    E : float
        Total energy of input photonic state.
    dE : float
        Energy difference between the two photons in the input state.
    """
    qs, D2, S2, S2in = smatrix.two_particle(s, chlsi=chlsi, chlso=chlso, E=E, dE=dE)

    Ee = np.array([.5 * (E - dE), .5 * (E + dE)])
    _, S10 = smatrix.one_particle(s, chlsi[0], chlso[0], Ee)
    _, S11 = smatrix.one_particle(s, chlsi[1], chlso[1], Ee)

    # denominator
    dFS2 = np.sum(np.abs(S10) ** 2 * np.abs(S11) ** 2)

    # Fourier transform
    FS2 = np.fft.fft(S2) * (qs[1] - qs[0])

    # frequencies
    freqs = np.fft.fftfreq(len(qs), d=qs[1] - qs[0])

    # add delta function contribution
    FS2 += np.exp(np.pi * 1j * E * freqs) * D2[0] + np.exp(np.pi * 1j * E * freqs) * D2[1]

    return FS2, dFS2


def FT(E, dE, qs, D2, S2):
    FS2 = np.fft.fft(S2) * (qs[1] - qs[0])
    times = np.fft.fftfreq(len(qs), d=qs[1] - qs[0])

    DFS2 = np.exp(np.pi * 1j * (E - dE) * times) * D2[0]
    DFS2 += np.exp(np.pi * 1j * (E + dE) * times) * D2[1]

    return times, FS2, DFS2

#     $$\      $$\           $$\
#     $$$\    $$$ |          \__|
#     $$$$\  $$$$ | $$$$$$\  $$\ $$$$$$$\
#     $$\$$\$$ $$ | \____$$\ $$ |$$  __$$\
#     $$ \$$$  $$ | $$$$$$$ |$$ |$$ |  $$ |
#     $$ |\$  /$$ |$$  __$$ |$$ |$$ |  $$ |
#     $$ | \_/ $$ |\$$$$$$$ |$$ |$$ |  $$ |
#     \__|     \__| \_______|\__|\__|  \__|

if __name__ == "__main__":

    # N = 6
    # m = bosehubbard.Model(
    #     omegas=[0]*N,
    #     links=[[i, (i+1) % N, 1] for i in xrange(N)],
    #     U=2)

    # # The Model
    U = 10
    model = Model(
        omegas=[0] * 2,
        links=[[0, 1, 1]],
        U=2 * U)

    couplings = []
    couplings.append(Coupling(channel=0, site=0, strength=.2))
    couplings.append(Coupling(channel=1, site=1, strength=.2))

    # The couplings
    couplings = []
    couplings.append(Coupling(channel=0, site=0, strength=.2))
    couplings.append(Coupling(channel=1, site=1, strength=.2))

    # The scattering setup
    se = Setup(model, couplings)

    # Let us calculate the t1matrix
    Es = Tmatrix.energies(0, 0)
    _, S100 = Smatrix.one_particle(se, 0, 0, Es)
    _, S101 = Smatrix.one_particle(se, 0, 1, Es)

    # Es = Tmatrix.energies(0, 0, WE=16, N=2048)
    E = 0
    dE = 0
    chlsi = [0, 0]
    chlso = [1, 1]

    qs = Tmatrix.energies(0, 0, 2048, 16)
    qs, D2, S2, S2in = Smatrix.two_particle(se, chlsi=chlsi, chlso=chlso, E=E, dE=dE, qs=qs)

    cplsi = [se.channel_coupling(c) for c in chlsi]
    cplso = [se.channel_coupling(c) for c in chlso]

    # # Fourier transform
    ts, FS2, DFS2 = FT(E, dE, qs, D2, S2)
    FS2 += DFS2
    FS2 /= (S101[512]) ** 2
    dt = ts[1] - ts[0]

    # # All the plots
    # # S1_{1,1}
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(Es, np.abs(S100) ** 2)
    ax1.set_ylabel(r'$S_{{{0},{1}}}^{{(1)}}$'.format(se.sites[0], se.sites[0]))

    # # S1_{0,1}
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(Es, np.abs(S101) ** 2)
    ax2.set_ylabel(r'$S_{{{0},{1}}}^{{(1)}}$'.format(se.sites[1], se.sites[0]))

    # # S2
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(Es, np.abs(S2), label=r"$S^{(2)}$")
    ax3.plot(Es, np.abs(S2in), label=r"$S^{(2)}_{in}$")
    ax3.plot(Es, np.abs(S2 - S2in), label=r"$S^{(2)}_{ni}$")
    ax3.set_ylabel(r'$S^{{(2)}}_{{{0},{1};{2},{3}}}$'.format(*([c.site for c in cplso + cplsi])))
    plt.legend()

    # # Fourier(S2)
    ax4 = plt.subplot(2, 2, 4)
    ax4.bar(ts - dt / 2, np.abs(FS2) ** 2, dt)
    ax4.set_ylabel(r'$|\mathcal{{F}} S^{{(2)}}_{{{0},{1};{2},{3}}}|^2$'.format(*([c.site for c in cplso + cplsi])))
    # ax4.set_xlim([-4, 4])s

    plt.suptitle(r'$\Gamma_{{{0}}} = {1}$, $\Gamma_{{{2}}} = {3}$'.format(se.sites[0], se.gs[0], se.sites[1], se.gs[1]))

    fig.set_tight_layout(True)
    plt.show()
