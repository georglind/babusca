from __future__ import print_function, division
import numpy as np
import scattering as scat
import smatrix
import matplotlib.pyplot as plt

# N = 12
# m = scat.Model(
#     omegas=[0]*N,
#     links=[[i, i+1, -1] for i in xrange(N-1)],
#     U=[0]*N)

# ns1 = m.numbersector(1)
# H1 = ns1.hamiltonian

# W, V = np.linalg.eig(H1.toarray())

# g = .1  #1/np.sqrt(np.pi)
# Sigma = np.zeros((N, N), dtype=np.complex128)
# Sigma[0, 0] = Sigma[N-1, N-1] = - 1j*np.pi*g**2

# print('Sigma')
# print(np.diag(Sigma))

# Sigm = V.conj().T.dot(Sigma).dot(V)
# # Sigm[np.abs(Sigm) < 1e-8] = 0

# print()
# print("V'SigmaV")
# print(np.imag(Sigm))

# channels = []
# channels.append(scat.Channel(site=0, strength=g))
# channels.append(scat.Channel(site=N-1, strength=g))

# s = scat.Setup(m, channels)
# E = s.eigenenergies(1)

# print()
# print('lambdas')
# print(E)


# plt.plot(np.sort(np.imag(E)))
# plt.plot(np.sort(np.diag(np.imag(Sigm))))
# plt.show()
