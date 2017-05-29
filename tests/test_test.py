from __future__ import print_function, division
import numpy as np
import scattering as scat
import smatrix
import matplotlib.pyplot as plt
from scipy import interpolate

# def averager(vals, navg):
#     nvals = len(vals)
#     outs = np.zeros((nvals-navg,), dtype=vals.dtype)
#     for i in xrange(nvals-navg):
#         outs[i] = np.sum(vals[i:(i+navg)])
#     return outs

# # interpolate.splrep(x, y, k=3, s=1000)

# ve = .5

# N = 100
# m = scat.Model(
#     omegas=[ve]*N,
#     links=[[i, i+1, 1] for i in xrange(N-1)],
#     U=[0]*N)

# c = np.sqrt(1)/np.sqrt(np.pi)
# qs = np.linspace(-2+ve, 2+ve, 1024+1)
# couplings = []
# couplings.append(scat.Channel(site=0, strength=c))
# couplings.append(scat.Channel(site=N-1, strength=c))

# s = scat.Setup(m, couplings)

# qs, S1 = smatrix.one_particle(s, 0, 1, qs)

# ff = 5
# # SS = averager(np.abs(S1)**2, ff)

# tck = interpolate.splrep(qs, np.abs(S1)**2, k=3, s=22)
# SS = interpolate.splev(qs, tck, der=0)
# # print(len(SS))
# plt.plot(qs, np.abs(S1)**2)
# plt.plot(qs, SS)
# plt.show()
