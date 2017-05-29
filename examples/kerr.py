from __future__ import division, print_function
import numpy as np

# import kerr
import figures
import generators.chain as chain


def figure1():
    Us = [.5, 1, 2, 10]
    ses = [chain.UniformChain(1, [0, 0], 0, [], Ui) for Ui in Us]

    for i in xrange(len(Us)):
        ses[i].label = r'${}$'.format(Us[i])

    Es = np.linspace(-5 / 2, 15 / 2, 1024)

    figures.g2_coherent(ses, (0, 0), (1, 1), Es, 2 * Es, ses[0].directory(), offset=0, ylims=[[1e-1, 2], [1e-1, 1e1]])


def figure2():

    U = 1
    se = chain.UniformChain(1, [0, 0], 0, [], U)

    Es = np.linspace(-5, 15, 1024)
    taus = np.linspace(0, 2, 1024)

    figures.g2_coherent_tau(se, (0, 0), (1, 1), Es, taus, directory=se.directory(), ticks=[0.75, 0.9, 1, 1.1, 1.25], title="for $U = \Gamma$")


def figure3():

    U = 10
    se = chain.UniformChain(1, [0, 0], 0, [], U)

    Es = np.linspace(-5, 15, 1024)
    taus = np.linspace(0, 2, 1024)

    figures.g2_coherent_tau(se, (0, 0), (1, 1), Es, taus, directory=se.directory(), ticks=[0, 0.5, 1, 1.5, 2], title="for $U = 10 \Gamma$")

if __name__ == "__main__":

    figure1()
    # figure2()
    # figure3()
