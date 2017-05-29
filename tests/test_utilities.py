from __future__ import division, print_function
import sys
import numpy as np
sys.path.append('/Users/kim/Science/Software/bosehubbard/bosehubbard')
sys.path.append('../')
import bosehubbard
import utilities as util


def test_create_boson():
    basis0 = bosehubbard.Basis(5, 0)
    basis1 = bosehubbard.Basis(5, 1)

    state0 = np.zeros((basis0.len,))
    state0[0] = 1

    state1 = util.create(0, state0, basis0, basis1)

    assert(np.isclose(np.abs(state1[0]), 1))


def test_annihilate_boson():
    basis0 = bosehubbard.Basis(5, 0)
    basis1 = bosehubbard.Basis(5, 1)

    state1 = np.zeros((basis1.len,))
    state1[0] = 1

    state0 = util.annihilate(0, state1, basis1, basis0)
    assert(np.isclose(np.abs(state0[0]), 1))


def test_annihilate_boson_create_boson():
    basis4 = bosehubbard.Basis(5, 4)
    basis5 = bosehubbard.Basis(5, 5)

    state4 = np.random.rand(basis4.len,)

    state5 = util.create(0, state4, basis4, basis5)
    nstate4 = util.annihilate(0, state5, basis5, basis4)

    assert(np.allclose((basis4.vs[:, 0]+1)*state4, nstate4))


def test_create_boson_annihilate_boson():
    basis4 = bosehubbard.Basis(5, 4)
    basis5 = bosehubbard.Basis(5, 5)

    state5 = np.random.rand(basis5.len,)

    state4 = util.annihilate(0, state5, basis5, basis4)
    nstate5 = util.create(0, state4, basis4, basis5)

    assert(np.allclose(basis5.vs[:, 0]*state5, nstate5))


if __name__ == '__main__':

    test_create_boson_annihilate_boson()
