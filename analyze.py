from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import g2
import views


def non_interacting_fermions_or_bosons(fermions, bosons):
    """
    Compares two-particle states to "fermionic" (Tonks-Girardeaux) states
    and pure non-interacting bosonic states.
    """
    E2, ii2l, ii2r = fermions.eigenbasis(2)
    si = np.argsort(np.real(E2))

    E2 = E2[si]
    ii2l = ii2l[:, si]
    ii2r = ii2r[:, si]

    F2, ni2l, ni2r = bosons.eigenbasis(2)
    sn = np.argsort(np.real(F2))

    F2 = F2[sn]
    ni2l = ni2l[:, sn]
    ni2r = ni2r[:, sn]
    N2 = len(E2)

    E1, psi1l, psi1r = fermions.eigenbasis(1)
    N1 = len(E1)

    fer11l = np.zeros((N2, N2), dtype=np.complex128)
    bos11l = np.zeros((N2, N2), dtype=np.complex128)

    EA = np.zeros((N2,), dtype=np.complex128)
    fi = []
    k = 0
    for i in xrange(N1):
        for j in xrange(i + 1):
            A = np.outer(psi1l[:, i], psi1l[:, j])
            EA[k] = E1[i] + E1[j]

            B = A - np.triu(A.T)
            iu, ju = np.triu_indices(B.shape[0])
            fer11l[:, k] = B[iu, ju]
            if i != j:
                fi.append(k)

            C = A + np.triu(A.T, 1)
            D = np.diag(C)
            C += (np.sqrt(2) - 1) * np.diag(D)
            iu, ju = np.triu_indices(C.shape[0])
            bos11l[:, k] = C[iu, ju]

            k += 1

    Ef = EA[fi]
    fer11l = fer11l[:, fi]

    s1 = np.argsort(np.real(Ef))
    Ef = Ef[s1]
    fer11l = fer11l[:, s1]

    s1 = np.argsort(np.real(EA))
    Eb = EA[s1]
    bos11l = bos11l[:, s1]

    olpa = ii2r.conj().T.dot(fer11l)
    olpb = ni2r.conj().T.dot(bos11l)

    plt.subplot(2, 2, 1)
    plt.plot(np.real(Ef), label='fermions')
    plt.plot(np.real(E2), label=r'$U \gg t$')
    plt.ylim([np.real(Ef[0]) - .5, np.real(Ef[-1]) + .5])
    plt.xlim([0, fer11l.shape[1] - 1])
    plt.ylabel('energies')
    plt.legend(loc='lower right')

    plt.subplot(2, 2, 2)
    plt.plot(Eb)
    plt.plot(F2)
    plt.xlim([0, N2])
    plt.ylabel('energies')

    plt.subplot(2, 2, 3)
    plt.pcolor(np.abs(olpa).T, vmin=0, vmax=1)
    plt.ylim([0, fer11l.shape[1] - 1])
    plt.xlim([0, N2])
    plt.ylabel('fermion-like')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.pcolor(np.abs(olpb).T, vmin=0, vmax=1)
    plt.xlim([0, N2])
    plt.ylim([0, N2])
    plt.ylabel('bosonic')
    plt.colorbar()

    plt.show()


def g2_coherent_tau0(ses, savedir, Es=None, **kwargs):
    # list of chain systems
    if Es is None:
        Es = np.linspace(-40, 40, 1024)  # energies

    # check out g2
    views.g2_coherent_tau0(ses, (0, 0), (1, 1), Es, savedir, **kwargs)
    views.g2_coherent_tau0(ses, (0, 0), (0, 0), Es, savedir, **kwargs)
    views.g2_coherent_tau0(ses, (0, 0), (0, 1), Es, savedir, **kwargs)
    # views.g2_coherent_tau0(ses, (0, 0), (1, 0), Es, savedir)


def phi2_coherent_tau0(ses, savedir, Es=None):
    if Es is None:
        Es = np.linspace(-40, 40, 1024)  # energies

    # check out g2
    views.phi2_coherent_tau0(ses, (0, 0), (1, 1), Es, savedir)
    views.phi2_coherent_tau0(ses, (0, 0), (0, 0), Es, savedir)
    views.phi2_coherent_tau0(ses, (0, 0), (0, 1), Es, savedir)
    # views.phi2_coherent_tau0(ses, (0, 0), (1, 0), Es, savedir)


def g2_coherent_tau(se, savedir, Es=None, taus=None, **kwargs):
    # contruct system

    if Es is None:
        Es = np.linspace(-40, 40, 1024)  # energies
    if taus is None:
        taus = np.linspace(0, 20, 512)  # time/position differences

    views.g2_coherent(se, (0, 0), (1, 1), Es, taus, savedir, **kwargs)
    views.g2_coherent(se, (0, 0), (0, 1), Es, taus, savedir, **kwargs)

    views.g2_coherent(se, (0, 0), (0, 0), Es, taus, savedir, **kwargs)


def phi2_coherent_tau(se, savedir, Es=None, taus=None):
    # contruct system

    if Es is None:
        Es = np.linspace(-40, 40, 1024)  # energies
    if taus is None:
        taus = np.linspace(0, 20, 512)  # time/position differences

    views.phi2_coherent(se, (0, 0), (1, 1), Es, taus, savedir)
    # views.g2_coherent(se, (0, 0), (0, 0), Es, taus, savedir)


def g2_fock_tau(se, savedir, Es=None, taus=None):

    if Es is None:
        Es = np.linspace(-40, 40, 1024)  # energies
    if taus is None:
        taus = np.linspace(0, 20, 512)  # time/position differences

    views.g2_fock(se, (0, 0), (1, 1), Es, 0, taus=taus, directory=savedir)
    views.g2_fock(se, (0, 1), (0, 1), Es, 0, taus=taus, directory=savedir)
    views.g2_fock(se, (0, 1), (0, 0), Es, 0, taus=taus, directory=savedir)


def nonlinear_transmittance(se, savedir, Us=None, Es=None):
    """
    Nonlinear stuff.
    """
    # systems
    ses = [se.clone(U=Ui) for Ui in Us]

    # energies
    if Es is None:
        Es = np.linspace(-20, 60, 1024)

    # extra = 'asym{}'.format(se.gs[0]/se.gs[1]) if gs[1] > 0 else 'asyminfty'

    views.nonlinear_transmittance(ses, 1, 1, Es, savedir)
    views.nonlinear_transmittance(ses, 0, 0, Es, savedir)
    views.nonlinear_transmittance(ses, 0, 1, Es, savedir)
    views.nonlinear_transmittance(ses, 1, 0, Es, savedir)


def movie(se0, ses, Es, Us, i=0):
    """
    Produces movieframes for various values of U.
    """
    # systems
    v0 = {'E2': None, 'phi2': None}
    v0['E2'], _, _ = se0.eigenbasis(2)
    v0['phi2'] = np.abs(g2.coherent_state_tau0(se0, (0, 0), (1, 1), Es)[1])

    for se in ses:
        f = movie_frame(se, v0, Es)
        plt.savefig(se0.directory() + 'frame{0:03d}.png'.format(i), dpi=200)
        plt.close(f)
        i += 1

    # ffmpeg -framerate 10/1 -start_number 000 -i frame%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4


def movie_frame(se, se0, Es):
    """
    Produces a movie-frame that shows two-particle eigenvalues along with the
    intensity-intensity correlation function g2.
    """
    f, (ax1, ax2) = plt.subplots(2, figsize=(6, 5))
    # energies
    E2, psi2l, psi2r = se.eigenbasis(2)

    vals = np.abs(doubly_occupied_weight(se, psi2l, psi2r))

    ax1.scatter(np.real(se0['E2']), np.imag(se0['E2']), 3, color='#bbbbbb')
    ax1.scatter(np.real(E2), np.imag(E2), 14, color='black')
    ax1.scatter(np.real(E2), np.imag(E2), 12, color=cm.viridis(vals[0]))

    ax1.set_xlabel(r'$\Re(\lambda^{(2)})$')
    ax1.set_ylabel(r'$\Im(\lambda^{(2)})$')
    ax1.set_xlim([np.min(Es), np.max(Es)])
    ax1.set_ylim([-2., .1])
    ax1.set_title(se.label)

    # phi2
    chlsi = (0, 0)
    chlso = (1, 1)
    phi2s = np.abs(g2.coherent_state_tau0(se, chlsi, chlso, Es)[1])
    ax2.semilogy(Es, se0['phi2'], linewidth=1, color="#bbbbbb")
    ax2.semilogy(Es, np.abs(phi2s), label=se.label, linewidth=1.5)
    ax2.set_ylabel(r'${typ}_{{{ins[0]}{ins[1]},{outs[0]}{outs[1]}}}$'.format(
        typ="\phi",
        ins=[c + 1 for c in chlsi],
        outs=[c + 1 for c in chlso]))
    ax2.set_xlabel(r'$\delta^{(2)}/\Gamma$')
    ax2.set_xlim([np.min(Es), np.max(Es)])
    ax2.set_ylim([1e-6, 2])
    plt.tight_layout()

    return f


def doubly_occupied_weight(se, psi2l, psi2r):
    """
    Return the weight of doublonic states within each two-particle state
    """
    basis = se.model.numbersector(2).basis.vs
    idx = np.where(np.any(basis == 2, axis=1))

    vals = np.sum(psi2l[idx, :].conj() * psi2r[idx, :], axis=1)

    return vals


def two_particle_state(psil, psir, basis):
    """
    Return two-particle basis states formed from single-particle states
    """
    n = basis.shape[1]
    st = np.zeros((n, n))

    vals = psil.conj() * psir
    for i in xrange(len(vals)):
        bas = basis[i, :]
        pos = np.nonzero(bas)[0]
        if len(pos) == 1:
            st[pos[0], pos[0]] = vals[i]
        else:
            st[pos[0], pos[1]] = vals[i]
            st[pos[1], pos[0]] = vals[i]

    return st
