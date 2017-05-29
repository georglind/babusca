from __future__ import division, print_function
import numpy as np
import scattering as scat

from nose.tools import assert_equals


def test_Channel():
    """
    Test the coupling object
    """
    c = scat.Channel(sites=[1], strengths=[2.1])
    assert_equals(c.sites[0], 1)
    assert_equals(type(c.sites[0]), int)
    assert_equals(c.strengths[0], 2.1)


def test_Setup():
    """
    Test the scattering setup object
    """
    m = scat.Model([0], [], [2])
    c1 = scat.Channel(sites=[1], strengths=[2.1])
    s = scat.Setup(model=m, channels=[c1])

    assert_equals(s.strengths, s.gs)


def test_eigenvector_overlaps():
    """"
    Test the eigenbasis
    """
    m = scat.Model([0] * 6, links=[[c, c + 1] for c in xrange(5)], U=[2] * 6)
    c1 = scat.Channel(sites=[0], strengths=[1.783])
    c2 = scat.Channel(sites=[4], strengths=[0.85])
    se = scat.Setup(model=m, channels=[c1, c2])

    E2, psi2l, psi2r = se.eigenbasis(2)

    assert np.allclose(np.abs(psi2l.conj().T.dot(psi2r)), np.eye(len(E2))), \
        "Basis is not normalized"


def test_noninteracting_dimer_eigenstates():
    m = scat.Model([.0] * 2, [[0, 1, 1]], [0] * 2)
    channels = [
        scat.Channel(sites=[0], strengths=[1.]),
        scat.Channel(sites=[1], strengths=[1.]),
    ]
    sp = scat.Setup(m, channels)

    # One-particle eigenbasis
    E1, psi1l, psi1r = sp.eigenbasis(1)

    # Two-particle eigenbasis
    E2, psi2l, psi2r = sp.eigenbasis(2)

    B = np.zeros((3, 3), dtype=np.complex128)

    for i in xrange(2):
        for j in xrange(2):
            for m in xrange(2):
                for n in xrange(2):
                    if m == n:
                        B[m + n, i + j] += np.sqrt(2) * psi1r[m, i] * psi1r[n, j]
                    else:
                        B[m + n, i + j] += psi1r[m, i] * psi1r[n, j]

    B[:, 0] /= np.sqrt(2)
    B[:, 1] /= 2
    B[:, 2] /= np.sqrt(2)

    # unphase
    psi2r *= np.sign(np.real(psi2r[0, :]))

    B *= np.sign(np.real(B[0, :]))

    assert np.allclose(psi2r, B), \
        'Non-interacting dimer, single particle states do not coincide with two-particle states'


def test_noninteracting_dimer_eigenenergies():
    """Test"""
    m = scat.Model([0] * 2, [[0, 1, 1.1]], [0] * 2)
    channels = [
        scat.Channel(site=0, strength=1),
        scat.Channel(site=1, strength=1),
    ]
    sp = scat.Setup(m, channels)

    E1, _, _ = sp.eigenbasis(1)
    E2, _, _ = sp.eigenbasis(2)

    E12 = np.zeros((len(E2), ), dtype=np.complex128)
    for i in xrange(len(E1)):
        for j in xrange(len(E1)):
            E12[i + j] = E1[i] + E1[j]

    E12 = E12[np.argsort(np.real(E12))]
    E2 = E2[np.argsort(np.real(E2))]

    assert np.allcose(E2, E12), \
        'Non-interacting dimer, single particle energies do not coincide with the two-particle energies.'


def test_Setup_transitions():
    """
    Test the scattering transitions
    """
    m = scat.Model([0] * 6, links=[[c, c + 1] for c in xrange(5)], U=[2] * 6)
    c1 = scat.Channel(site=0, strength=1.)
    c2 = scat.Channel(site=4, strength=1.)
    se = scat.Setup(model=m, channels=[c1, c2])

    A10, A01 = [], []
    for c in xrange(se.model.n):
        A10.append(se.trsn(0, c, 1))
        A01.append(se.trsn(1, c, 0))

    A = np.zeros((se.model.n, se.model.n))
    for i in xrange(se.model.n):
        for j in xrange(se.model.n):
            A[i, j] = np.abs(A01[i].dot(A10[j])[0][0])

    # Test that A is orthogonal
    assert np.allclose(A, np.eye(se.model.n))

    A21, A12 = [], []
    for c in xrange(se.model.n):
        A21.append(se.trsn(1, c, 2))
        A12.append(se.trsn(2, c, 1))

    # Assert that A21 works
    for c in xrange(se.model.n):
        assert np.isclose(np.abs(A01[c].dot(A12[c]).dot(A21[c]).dot(A10[c])), 2), \
            'Transitions 2 -> 1 not properly normalized'

