from __future__ import division, print_function
import numpy as np
import bosehubbard   # model base
import graph         # forcedirectedgraph layout
import scipy.linalg as linalg
import scipy.sparse as sparse


#     $$\      $$\                 $$\           $$\
#     $$$\    $$$ |                $$ |          $$ |
#     $$$$\  $$$$ | $$$$$$\   $$$$$$$ | $$$$$$\  $$ |
#     $$\$$\$$ $$ |$$  __$$\ $$  __$$ |$$  __$$\ $$ |
#     $$ \$$$  $$ |$$ /  $$ |$$ /  $$ |$$$$$$$$ |$$ |
#     $$ |\$  /$$ |$$ |  $$ |$$ |  $$ |$$   ____|$$ |
#     $$ | \_/ $$ |\$$$$$$  |\$$$$$$$ |\$$$$$$$\ $$ |
#     \__|     \__| \______/  \_______| \_______|\__|

class Model(bosehubbard.Model):
    """
    Extended version of Bose-Hubbard model with caching of number-sectors.
    """
    def __init__(self, Es, links, Us, W=None):
        """
        Initiate our scattering structure.

        Parameters
        ----------
        Es : list
            List of onsite energies.
        links : list of lists
            List of links on the form [site_1, site_2, strength].
        Us : list
            Onsite interaction strengths
        """
        bosehubbard.Model.__init__(self, Es, links, Us, W)
        self.reset()

    def numbersector(self, nb):
        """
        Returns a specific particle number sector object based on this model.
        Cache the result for later retrieval.

        Parameters
        ----------
        nb : int
            Number of bosons in the given number sector
        """
        if nb not in self._cache['ns']:
            self._cache['ns'][nb] = bosehubbard.NumberSector(self.n, nb, model=self)

        return self._cache['ns'][nb]

    def reset(self):
        """
        Remove all cached sectors
        """
        self._cache = {'ns': {}}

    def draw(self, fig=None, ax=None):
        """
        Clever force directed plot of any graph.
        """
        g = graph.Graph(self.Es, None, self.links)
        g.forcedirectedlayout()
        g.plot(fig, ax)


#      $$$$$$\  $$\                                               $$\
#     $$  __$$\ $$ |                                              $$ |
#     $$ /  \__|$$$$$$$\   $$$$$$\  $$$$$$$\  $$$$$$$\   $$$$$$\  $$ |
#     $$ |      $$  __$$\  \____$$\ $$  __$$\ $$  __$$\ $$  __$$\ $$ |
#     $$ |      $$ |  $$ | $$$$$$$ |$$ |  $$ |$$ |  $$ |$$$$$$$$ |$$ |
#     $$ |  $$\ $$ |  $$ |$$  __$$ |$$ |  $$ |$$ |  $$ |$$   ____|$$ |
#     \$$$$$$  |$$ |  $$ |\$$$$$$$ |$$ |  $$ |$$ |  $$ |\$$$$$$$\ $$ |
#      \______/ \__|  \__| \_______|\__|  \__|\__|  \__| \_______|\__|

class Channel:
    """
    Coupling between a single channel and one or *more* sites.
    """
    def __init__(self, site=None, sites=None, strength=None, strengths=None, positions=None):
        """
        Initialize coupling object.

        Parameters
        ----------
        channel : int
            Channel index
        site : int
            Site index
        sites : list
            list of site indices for each coupling to this channel
        strength : float
            Coupling strength
        strengths : list
            list of coupling strengths for each coupling
        positions: list
            list of positions of each coupling coordinate
        """
        # set sites and strenths
        sites = np.atleast_1d(sites if site is None else site)
        strengths = np.atleast_1d(strengths if strength is None else strength)

        # set positions.
        if positions is None:
            positions = [0] * len(sites)

        positions = np.atleast_1d(positions)

        # indices of non-zero strength
        idx = strengths != 0

        # re-index everything
        self.sites = sites[idx]
        self.strengths = strengths[idx]
        self.positions = positions[idx]

        # number of couplings
        self.n = len(self.sites)

        # is the coupling local or quasi-local
        self.local = np.allclose(self.positions, self.positions[0] * np.ones((self.n, )), 1e-8)

        # if all couplings are local, simplify the results by fixing the positions to zero
        if self.local:
            self.positions = np.zeros((self.n, ), dtype=np.float64)

    def gtilde(self, phi=0):
        """
        Effective coupling strengths dressed by phase factors.

        Note that gtilde() explicitly implements the prefactor to b^dagger and **not** b.

        Parameters
        ----------
        phi : float
            The energy/wavenumber parameter in units of inverse length.
        """
        return self.strengths * np.exp(1j * phi * self.positions)

    @property
    def gs(self):
        """Alias for all coupling strengths."""
        return self.strengths

    @property
    def xs(self):
        """Alias for positions."""
        return self.positions


#      $$$$$$\             $$\
#     $$  __$$\            $$ |
#     $$ /  \__| $$$$$$\ $$$$$$\   $$\   $$\  $$$$$$\
#     \$$$$$$\  $$  __$$\\_$$  _|  $$ |  $$ |$$  __$$\
#      \____$$\ $$$$$$$$ | $$ |    $$ |  $$ |$$ /  $$ |
#     $$\   $$ |$$   ____| $$ |$$\ $$ |  $$ |$$ |  $$ |
#     \$$$$$$  |\$$$$$$$\  \$$$$  |\$$$$$$  |$$$$$$$  |
#      \______/  \_______|  \____/  \______/ $$  ____/
#                                            $$ |
#                                            $$ |
#                                            \__|

class Setup:
    """
    A complete quasi local scattering  Setup with
    scattering structure (model), channels, and parasitic couplings.

    This setup allow for a quasi-locally coupled scatterer.
    We employ the Markov approximation for propagation
    within the channels.

    We retain the phase acquired between coupling sites, as well as
    a certain directionality in terms of off-diagonal elements of the
    coupling Hamiltonian. In addition we describe dynamics within the
    scatterer exactly.
    """
    def __init__(self, model, channels, parasites=None):
        """
        Initialize the scattering setup.

        Parameters
        ----------
        model : Model object
            Describes the bosehubbard scattering centre
        channels : list of channels objects
            List of channels
        parasites : List of Coupling objects
            List of parasitic coupling objects
        """
        self.model = model
        self.channels = tuple(channels)
        self.parasites = tuple(parasites) if parasites is not None else ()

        # is the setup local?
        self.local = all([channel.local for channel in self.channels])

        # reset all caches
        self.reset(model=False)

    def reset(self, model=True):
        """Delete all caches."""
        if model:
            self.model.reset()
        self._cache = {'eigen': {}, 'trans': {}, 'trsn': {}, 'sigma': {}}

    def eigenbasis(self, nb, phi=0):
        """
        Calculates the generalized eigen-energies along with
        the left and right eigen-basis.

        Parameters
        ----------
        nb : int
            Number of bosons
        phi : float
            Phase factor for the relevant photonic state
        """
        phi = 0 if self.local else phi

        ckey = '{}-{}'.format(nb, phi)
        if ckey not in self._cache['eigen']:
            # generate number sector
            ns1 = self.model.numbersector(nb)

            # get the size of the basis
            ns1size = ns1.basis.len  # length of the number sector basis
            # G1i = xrange(ns1size)    # our Greens function?

            # self energy
            sigma = self.sigma(nb, phi)

            # Effective Hamiltonian
            H1n = ns1.hamiltonian + sigma

            # Complete diagonalization
            E1, psi1r = linalg.eig(H1n.toarray(), left=False)
            psi1l = np.conj(np.linalg.inv(psi1r)).T

            # check for dark states (throw a warning if one shows up)
            # if (nb > 0):
            #     Setup.check_for_dark_states(nb, E1)

            self._cache['eigen'][ckey] = (E1, psi1l, psi1r)

        return self._cache['eigen'][ckey]

    @staticmethod
    def check_for_dark_states(nb, Es):
        """Check for dark states, throws a warning if it finds one."""
        dark_state_indices = np.where(np.abs(np.imag(Es)) < 10 * np.spacing(1))

        if len(dark_state_indices[0]) == 0:
            return

        import warnings
        warnings.warn('The {} block contains {} dark state(s) with generalized eigenenergie(s): {}'.format(nb, len(dark_state_indices), Es[dark_state_indices]))

    def sigma(self, nb, phi=0):
        """
        Local and quasi-local self energy

        Parameters
        ----------
        nb : int
            number sector, number of bosons
        phi : float
            phase contribution in units of energy per length
        """

        # Local systems have no phases
        phi = 0 if self.local else phi

        # Cache the local part
        ckey = '{}-{}'.format(nb, phi)
        if ckey not in self._cache['sigma']:
            # cache the results
            self._cache['sigma'][ckey] = Setup.sigma_local(self.model, self.channels + self.parasites, nb)

        # Load local sigma from cache
        sigmal = self._cache['sigma'][ckey]

        # if it is only local: break off calculation here
        if self.local:
            return sigmal

        # generate the additional quasi-local contribution to the self energy
        # if nb == 0:
        #     sigmaql = np.zeros((1,))
        # else:
        sigmaql = Setup.sigma_quasi_local(self.model, self.channels, nb, phi)

        # return local and quasi-local contribution to the self-energy
        return sigmal + sigmaql

    @staticmethod
    def sigma_local(model, channels, nb):
        """
        Computes the local self-energy

        Parameters
        ----------
        model : Model object
            Model object
        channels: List of Channel objects
            Contains all channels (also )
        nb : int
            number of bosons/photons
        """
        Gams = np.zeros((model.n, model.n), dtype=np.complex128)

        # iterate over all sites
        for channel in channels:
            for n, sn in enumerate(channel.sites):
                # diagonal elements
                Gams[sn, sn] += - 1j * np.pi * np.abs(channel.strengths[n]) ** 2
                for m, sm in enumerate(channel.sites[(n + 1):]):
                    # off-diagonal elements
                    if channel.xs[sn] == channel.xs[sm]:
                        Gams[sn, sm] += - 1j * np.pi * np.conjugate(channel.strengths[m]) * channel.strengths[n]
                        Gams[sm, sn] += - 1j * np.pi * np.conjugate(channel.strengths[n]) * channel.strengths[m]

        # nb numbersector
        ns = model.numbersector(nb)

        # generate relevant hamiltonian
        Ski, Skj, Skv = ns.hopping_hamiltonian(ns.basis, Gams, ns.basis.vs)

        # construct dense matrix
        Sigma = sparse.coo_matrix((Skv, (Ski, Skj)), shape=[ns.basis.len] * 2).tocsr()

        return Sigma

    @staticmethod
    def sigma_quasi_local(model, channels, nb, phi=0):
        """
        Quasi-local self-energy

        Parameters
        ----------
        model : Model object
            Model object
        channels: List of Channel objects
            Contains all channels (also )
        nb : int
            number of bosons/photons
        phi : float
            phase contribution in units of energy per length
        """
        Gams = np.zeros((model.n, model.n), dtype=np.complex128)

        # iterate over all channels
        for channel in channels:
            # skip local channels
            if channel.local is True:
                continue

            # iterate all couplings to this channel
            for n, sn in enumerate(channel.sites):
                posn = channel.positions[n]
                gn = channel.strengths[n]

                # iterate all couplings to this channel
                for m, sm in enumerate(channel.sites):
                    posm = channel.positions[m]
                    gms = np.conjugate(channel.strengths[m])

                    # only one channel chirality contributes.
                    # I.e. "leave the model through one channel
                    # and return from a point further down that same channel".
                    if posn > posm:
                        Gams[sn, sm] += - 2 * 1j * np.pi * gms * gn * np.exp(1j * phi * (posn - posm))

        # nb numbersector
        ns = model.numbersector(nb)

        # generate relevant hamiltonian
        Ski, Skj, Skv = ns.hopping_hamiltonian(ns.basis, Gams, ns.basis.vs)

        # construct dense matrix
        Sigma = sparse.coo_matrix((Skv, (Ski, Skj)), shape=[ns.basis.len] * 2).tocsr()

        return Sigma

    def eigenenergies(self, nb, phi=0):
        """
        Return a list of eigenenergies in a given number sector.

        Parameter
        ---------
        nb : int
            Number of bosons in given number sector
        phi: float
            phase related to the photonic energy
        """
        # no phases for local setups
        phi = 0 if self.local else phi

        # cache
        ckey = '{}-{}'.format(nb, phi)
        if ckey not in self._cache['eigen']:
            self.eigenbasis(nb, phi)

        # load cached result
        return self._cache['eigen'][ckey][0]

    def transition(self, ni, channel, nf, phi=0):
        """
        Generalized transition matrix elements for a single channel

        Parameters
        ----------
        ni : int
            initial charge sector
        channel : int
            channel index
        nf : int
            final number sector
        phis : dict
            dict of phase parameters for coupling constants (g),
            incoming/initial photon number sector (i), and final
            photon number sector (f): {i: value, g: value, f: value}.
        """
        # no phases for local setups
        phi = 0 if self.local else phi

        # cache
        ckey = '{}-{}-{}-{}'.format(nf, channel, ni, phi)
        if ckey not in self._cache['trans']:

            # Effective coupling constant in front of b^\dagger
            gt = self.channels[channel].gtilde(phi)

            if nf < ni:
                gt = np.conj(gt)   # b = (b^dagger)^dagger

            gen = (gt[i] * self.trsn(ni, self.channels[channel].sites[i], nf, phi) for i in xrange(self.channels[channel].n))
            self._cache['trans'][ckey] = sum(gen)

        return self._cache['trans'][ckey]

        # gt = self.channels[channel].gtilde(phi)
        #  gen = (self.channels[channel].strengths[i] * self.trsn(ni, self.channels[channel].sites[i], nf) for i in xrange(len(self.channels[channel].sites)))
        #     self._cache['trans'][key] = sum(gen)

        # return np.sum([gt[i] * self.trsn(ni, self.channels[channel].sites[i], nf, phi) for i in xrange(self.channels[channel].n)])

    def trsn(self, ni, site, nf, phi=0):
        """
        Bare transition matrix elements in the sites basis.

        Parameters
        ----------
        ni : int
            initial charge sector
        site : int
            Model site index
        nf : int
            final number sector
        phis : dict
            dict of phase parameters for
            incoming/initial photon number sector (i), and final
            photon number sector (f): {i: value, f: value}.
        """
        # no phases in local setups
        phi = 0 if self.local else phi

        # cache
        ckey = '{}-{}-{}-{}'.format(nf, site, ni, phi)
        if ckey not in self._cache['trsn']:
            # initial
            nsi = self.model.numbersector(ni)
            Ei, psiil, psiir = self.eigenbasis(ni, phi)

            # final
            nsf = self.model.numbersector(nf)
            Ef, psifl, psifr = self.eigenbasis(nf, phi)

            # transition
            A = np.zeros((nsf.basis.len, nsi.basis.len), dtype=np.complex128)
            for i in xrange(nsi.basis.len):
                A[:, i] = psifl.conj().T.dot(
                    bosehubbard.transition(site, psiir[:, i], nsi.basis, nsf.basis)
                )

            self._cache['trsn'][ckey] = A

        return self._cache['trsn'][ckey]


# UTILITIES

def discrete_energies(E, dE, N=1024, WE=8.):
    """
    Discretization of scattering energies.

    Parameters
    ----------
    E : float
        Total two-particle energy
    dE : float
        Two-particle energy difference
    N : int
        Number of discretization points
    WE : float
        Half width of the scattering energy spectrum
    """
    # Discrete energies that contains the "elastic" points:
    # nu0=nu2 and nu0=nu3.
    NdE = np.ceil(N / 2 / (dE / 2 + WE) * dE / 2)
    Lq = N / 2 / NdE * dE / 2 if np.abs(dE) > 0 else WE
    qs = np.linspace(-Lq, Lq, N, endpoint=False) + E / 2

    return qs
