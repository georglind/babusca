# File: bosehubbard.py
#
# Efficient but simplistic approach to generating the
# many-body Hamiltonian for Bose-Hubbard models.
# Can be used for exact diagonalization.
#
# Code released under a MIT license, by
# Kim G. L. Pedersen, 2015
# (unless otherwise noted in the function description)
#
# Waiver: No guarantees given. Please use **completely** at your own risk.

import numpy as np
from scipy.special import binom
import scipy.sparse as sparse


class Model:
    """
    Defines some Bose-Hubbard model by specifying the onsite energies (es),
    the links between sites on the network, and
    the onsite interaction strength (U)

    Parameters
    ----------
    es : list
        Onsite energies
    links : list of lists
        Each hopping is of the form [from, to, amplitude].
    U : list
        Interaction strength
    """
    def __init__(self, Es, links, Us, W=None):
        self.Es = np.array(Es)
        self.links = links
        self.Us = np.array(Us, dtype=np.float64)
        self.W = W

        self.n = len(Es)

        # cache
        self._cache = dict()

    @property
    def hopping(self):
        """
        The single particle hopping Hamiltonian
        """
        H0 = np.zeros([self.n] * 2, np.complex128)

        for link in self.links:
            H0[link[0], link[1]] = link[2] if len(link) > 2 else -1

        return H0 + H0.conj().T

    def numbersector(self, nb):
        """
        Returns a specific particle number sector object based on this model.
        """
        return NumberSector(self.n, nb, model=self)


class NumberSector:
    """
    Defines a specific particle number sector of a given Bose-Hubbard model.
    """
    def __init__(self, N, nb, model=None):
        self.N = N
        self.nb = nb
        self.basis = Basis(N, nb)
        if model is not None:
            self.model = model

    @property
    def hamiltonian(self):
        # cache hamiltonian
        if not hasattr(self, '_hamiltonian') or not hasattr(self._hamiltonian, 'shape'):
            self._hamiltonian = NumberSector.generate_hamiltonian(self.model, self.basis)

        return self._hamiltonian

    @staticmethod
    def generate_hamiltonian(m, basis):
        """
        Generates the (sparse) Hamiltonian

        Parameters
        ----------
        basis : Basis object
            Full basis for this specific number sector.
        """
        nbas = basis.len

        Us = m.Us
        if m.W is not None:
            Us = m.W + np.diag(m.Us)

        HDi = np.arange(nbas)
        HD = NumberSector.onsite_hamiltonian(m.Es, basis.vs) \
            + NumberSector.interaction_hamiltonian(Us, basis.vs)
        Hki, Hkj, Hkv = NumberSector.hopping_hamiltonian(basis, m.hopping, basis.vs)

        return sparse.coo_matrix((Hkv, (Hki, Hkj)), shape=(nbas, nbas)).tocsr() \
            + sparse.coo_matrix((HD, (HDi, HDi)), shape=(nbas, nbas)).tocsr()

    @staticmethod
    def onsite_hamiltonian(es, states):
        """
        Onsite hamiltonian

        Parameters
        ----------
        es : ndarray
            List of onsite energies.
        states : ndarray
            List of many-body states.
        """
        return states.dot(es)

    @staticmethod
    def hopping_hamiltonian(basis, H0, states=None):
        """
        Hopping Hamiltonian expressed in the many-particle basis

        Parameters
        ----------
        basis : Basis object
            The relevant basis object
        H0 : ndarray
            Single particle operator
        states : ndarray
            Many-body basis states in number representation.
        """
        if states is None:
            states = basis.vs

        H1s, H2s, Hvs = [], [], []

        # we deal with diagonal (identical to onsite_hamiltonian) elements first
        H0d = np.diag(H0)

        if np.any(np.abs(H0d) > 0):
            H0 = H0 - np.diag(H0d)  # remove diagonal entries
            H0i = np.arange(basis.len).tolist()

            # add this to the hamiltonian
            H1s += H0i
            H2s += H0i
            Hvs += (states.dot(H0d)).tolist()

        # then off diagonal (proper hopping) elements
        for i in range(H0.shape[0]):
            js = np.nonzero(states[:, i])[0]  # affected states
            nj = len(js)

            ls = np.nonzero(H0[i, :])[0]  # relevant hoppings
            nl = len(ls)

            ks = np.zeros((nj * nl,))  # storing result states
            vs = np.zeros((nj * nl,), dtype=H0.dtype)  # storing result states

            for k, l in enumerate(ls):
                nstates = states[js, :]
                nstates[:, i] -= 1  # remove one element
                nstates[:, l] += 1  # add it here
                ks[k * nj:(k + 1) * nj] = basis.index(nstates)  # the new states
                vs[k * nj:(k + 1) * nj] = H0[i, l] * np.sqrt(states[js, i] * (states[js, l] + 1))

            H1s += np.tile(js, nl).tolist()
            H2s += ks.tolist()
            Hvs += vs.tolist()

        return H1s, H2s, Hvs

    @staticmethod
    def interaction_hamiltonian(HU, states):
        """
        Return the interaction energy for each state.

        Parameters
        ----------
        HU : ndarray
            Interaction strength(s)
            Can be 2-dim with offsite interaction strength
        states : ndarray
            Many-body basis states in the number representation.
        """
        if len(HU.shape) == 1:
            # Only onsite interaction
            return .5 * np.sum(np.dot(states * (states - 1), HU[:, None]), axis=1)
        else:
            U = np.diag(HU)
            W = HU - np.diag(U)

            # also offsite interaction
            return .5 * np.sum(np.dot(states * (states - 1), U[:, None]), axis=1) + .5 * np.sum(states.dot(W) * states, axis=1)


# Functions for transitioning between different number sectors of a
# Bose-Hubbard model

def creation_operator(i, basis0, basis1):
    """
    Create a boson on site <i>

    Parameters
    ----------
    i : int
        Site index
    basis0 : list
        Initial basis
    basis1 : list
        Final basis
    """
    index0 = np.arange(basis0.len)

    mbasis = np.copy(basis0.vs)
    mbasis[:, i] += 1
    index1 = basis1.index(mbasis)

    return sparse.coo_matrix((np.sqrt(mbasis[:, i]), (index1, index0)), shape=[basis1.len, basis0.len]).tocsr()


def annihilation_operator(i, basis1, basis0):
    """
    Annihilate a boson on site <i>

    Parameters
    ----------
    i : int
        Site index
    basis1 : list
        Initial basis
    basis0 : list
        Final basis
    """
    return creation_operator(i, basis0, basis1).T


def create(i, state0, basis0, basis1):
    """
    Create a boson on site <i>

    Parameters
    ----------
    i : int
        Site index
    state0 : ndarray
        Initial state
    basis0 : list
        Initial basis
    basis1 : list
        Final basis
    """
    mbasis = np.copy(basis0.vs)
    mbasis[:, i] += 1

    state1 = np.zeros((basis1.len,), dtype=np.complex128)

    index1 = basis1.index(mbasis)

    state1[index1] = state0 * np.sqrt(mbasis[:, i])

    return state1


def annihilate(i, state1, basis1, basis0):
    """
    Annihilate a boson on site i, and return result in the standard basis0
    """
    js = np.nonzero(basis1.vs[:, i])[0]

    mbasis = basis1.vs[js, :]
    mbasis[:, i] -= 1

    state0 = np.zeros((basis0.len,), dtype=np.complex128)    # new state
    index0 = basis0.index(mbasis)  # index of the resulting states

    state0[index0] = state1[js] * np.sqrt(basis1.vs[js, i])

    return state0


def transition(i, statei, basisi, basisf):
    """
    Transition from one number sector to the other by removing a boson.
    """
    if basisi.nb > basisf.nb:
        return annihilate(i, statei, basisi, basisf)
    else:
        return create(i, statei, basisi, basisf)


def create_elements(psil, psir, n1basis, sites, n0basis):
    """
    Returns matrix elements
    < psil | bd_i | psir >

    Parameters
    ----------
    sites : list
        List of sites to calculate matrix elements for.
    psil : ndarray
        Left eigenvectors
    psir : ndarray
        Right eigenvectors
    n0basis : Basis
        Number basis for 0 bosons
    n1basis : Basis
        Number basis for 1 boson states
    """
    bds = np.zeros((len(sites), 2, psil.shape[0]), dtype=np.complex128)

    for i, site in enumerate(sites):
        # add one boson on the relevant site and find
        # the corresponding state vector written in the n=1 number basis
        init = create(site, np.array([1]), n0basis, n1basis)

        # calculate the overlap with the left and right eigenbasis in the
        # n=1 number basis
        bds[i, 0, :] = init.dot(psil)    # left eignbasis
        bds[i, 1, :] = init.dot(psir)    # right eigenbasis

    return bds.conj()


class Basis:
    """
    Many-body basis of specific <nb> charge state on a lattice with <N> sites.
    """
    def __init__(self, N, nb):
        """
        Parameters
        ----------
        N : int
            Number of sites
        nb : int
            Number of bosons
        """
        self.N = N  # number of sites
        self.nb = nb  # number of bosons

        self.len = Basis.size(N, nb)
        self.vs = Basis.generate(N, nb)
        self.hashes = Basis.hash(self.vs)
        self.sorter = Basis.argsort(self.hashes)

    def index(self, state):
        """
        Find the index of the state in self.basis.

        Parameters
        ----------
        state : ndarray
            One or more states.
        """
        return Basis.stateindex(state, self.hashes, self.sorter)

    @staticmethod
    def size(N, nb):
        """
        Return the size of the boson many-body basis.

        Parameters
        ----------
        N : int
            Number of sites
        nb : int
            Number of bosons
        """
        return int(binom(nb + N - 1, nb))

    @staticmethod
    def stateindex(state, hashes, sorter):
        """
        Converts state to hash and searches for the hash among hashes,
        which are sorted by the sorter list.

        Parameters
        ----------
        state : ndarray
            An array of one or more states
        hashes : ndarray
            List of hashes so search among
        sorter : ndarray
            Sorting indicies which sorts hashes
            (generated from Basis.argsort).
        """
        key = Basis.hash(state)
        return sorter[np.searchsorted(hashes, key, sorter=sorter)]

    @staticmethod
    def generate(N, nb):
        """
        Generate basis incrementally based on the method described in e.g.
        http://iopscience.iop.org/article/10.1088/0143-0807/31/3/016

        Parameters
        ----------
        N : int
            Number of sites
        nb : int
            Number of bosons
        """
        states = np.zeros((Basis.size(N, nb), N), dtype=int)

        states[0, 0] = nb
        ni = 0  # init
        for i in range(1, states.shape[0]):

            states[i, :N - 1] = states[i - 1, :N - 1]
            states[i, ni] -= 1
            states[i, ni + 1] += 1 + states[i - 1, N - 1]

            if ni >= N - 2:
                if np.any(states[i, :N - 1]):
                    ni = np.nonzero(states[i, :N - 1])[0][-1]
            else:
                ni += 1

        return states

    @staticmethod
    def hash(states):
        """
        Hash function as given in:
        http://iopscience.iop.org/article/10.1088/0143-0807/31/3/016

        Parameters
        ----------
        states : ndarray
            List of states (that will be hashed)
        """
        n = states.shape[1] if len(states.shape) > 1 else len(states)
        ps = np.sqrt(lowest_primes(n))
        return states.dot(ps)

    @staticmethod
    def argsort(hashes, algorithm='quicksort'):
        """
        Argsort our hashes for searching, using e.g. quicksort.
        """
        return np.argsort(hashes, 0, algorithm)


def lowest_primes(n):
    """
    Return the lowest n primes

    Parameters
    ----------
    n : int
        Number of primes to return
    """
    return primes(n ** 2)[:n]


def primes(upto):
    """
    Prime sieve below an <upto> value.
    Copied from http://rebrained.com/?p=458

    Parameters
    ----------
    upto : int
        Find all primes [less than or equal] to this limit.
    """
    primes = np.arange(3, upto + 1, 2)

    isprime = np.ones(int((upto - 1) / 2), dtype=bool)

    for factor in primes[:int(np.sqrt(upto))]:
        if isprime[int((factor - 2) / 2)]:
            isprime[int((factor * 3 - 2) / 2)::factor] = 0

    return np.insert(primes[isprime], 0, 2)
