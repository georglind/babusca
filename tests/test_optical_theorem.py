from __future__ import division, print_function
import sys
sys.path.append('../')
import numpy as np
import scattering
import smatrix
import matplotlib.pyplot as plt

# model
N = 2
m = scattering.Model(
    omegas=[0] * N,
    links=[(0, 1, .5)],
    U=[5] * N)

# channel
channels = []
channels.append(scattering.Channel(site=0, strength=1))
channels.append(scattering.Channel(site=1, strength=1))

# setup
s = scattering.Setup(m, channels)

# input state
E = 0
dE = 4.38821403174
chlsi = [0, 0]

# output state
chlsos = [[0, 0], [0, 1], [1, 0], [1, 1]]
qs = np.linspace(-.1, .1, 8192, endpoint=False)

d = 120
# qs = np.linspace(-d, d, 8192, endpoint=False)
qs, iq0 = scattering.energies(E, dE, N=2*8192, WE=2 * d)

plt.figure(figsize=(6, 6))
D2s, S2s = [], []
for c, chlso in enumerate(chlsos):
    qs, D2, S2, S2in = smatrix.two_particle(s,
                                            chlsi=chlsi,
                                            chlso=chlso,
                                            E=E,
                                            dE=dE,
                                            qs=qs)

    D2s.append(D2)
    S2s.append(S2)
    plt.subplot(2, 2, c + 1)
    plt.plot(qs, np.real(S2))
    plt.plot(qs, np.imag(S2))

for c, chlso in enumerate(chlsos):
    dsums = [np.sum(np.abs(D2s[c]) ** 2) for c in xrange(4)]
    ssums = [np.sum(np.abs(S2s[c]) ** 2) * (qs[1] - qs[0]) for c in xrange(4)]
    dssums = [2 * np.real(np.sum(D2s[c]).conj() * S2s[c][iq0]) for c in xrange(4)]


print(dsums)
print(ssums)
print(dssums)

print(np.sum(dsums))
print(np.sum(ssums) + np.sum(dssums))

plt.show()



# iq0 = np.where(qs == E / 2)[0][0]

# dsums = [np.sum(np.abs(D2s[c]) ** 2) for c in xrange(4)]
# ssums = [np.sum(np.abs(S2s[c]) ** 2) * (qs[1] - qs[0]) for c in xrange(4)]
# dssums = [2 * np.real(np.sum(D2s[c]).conj() * S2s[c][iq0]) for c in xrange(4)]
