# babusca

Scattering theory for few-photon transport through Bose-Hubbard lattices.

Library capable for calculation first and second order coherence function for Bose-Hubbard lattices coupled to chiral channels. On current equipment babusca can practically cope with system of up to 120-150 sites.

The code is well documented, and several examples of simple systems are provided in the examples folder.

Please do not hesitate to contact me for questions or comments. 

## Contructing the scatterer

The Bose-Hubbard scatterer is specificed using the scattering library.

Say, we want to construct a three-site model with:
- Onsite energies at `E0 = +1e9+1`, `E1 =1e9`, `E2 = 1e9-1`.
- Site 0 and 1 are coupled by a strength `t01 = 2.5`, and site 2 and 3 by the strength `t12=1.5`.
- Onsite photon-photon interactions `U0 = 1`, `U1 = 1`, and `U2 = 1`.

```python
# construct scatterer
mo = scattering.Model(Es=[1e9 + 1, 1e9, 1e9 -1], links=[(0, 1, 2.5), (1, 2, 1.5)], Us = [1, 1, 1])
```

One may also include off-site two-body interactions by providing a symmetric `W` array, with entries `W[i,j]` detailing the interaction strength between photons on site `i` and site `j`.

It is also possible to visualize the scatterer geometry by using the `graph` method.

```python
 # plot the scatterer geometry
 mo.graph()
```

## Constructing the chiral channels

Say, we want to connect our three site model to a chiral channel,
- From site `j = 0` of the model.
- At a point `x = 0` on the channel.
- And with a strength `g = 1/numpy.sqrt(numpy.pi)`.

```python
import numpy
import scattering

# construcing a channel
ch0 = scattering.Channel(site=0, position=0, strength=1/numpy.sqrt(numpy.pi))
```

We can also attach an additional channel. Let us this time deal with a channel, which couples at two points,
- From `x = 0` to site `j = 1` with a strength `g = 1/numpy.sqrt(2*numpy.pi)`.
- And from `x = np.pi / 1e9` to site `j = 2` with a strength `g = 1/numpy.sqrt(2*numpy.pi)`.

```python
# constructing quasi-locally coupled channel
ch1 = scattering.channel(sites=[1, 2], strengths=[1/numpy.sqrt(2*numpy.pi)]*2, positions=[0, np.pi / 1e9])
```

## Constructing the full scattering setup

We concoct the channels and the model into a complete scattering setup.
```python
# Concoct scatterer
se = scattering.Setup(model=mo, channels=[ch0, ch1])
```

It is also possible to add additional decay channels by passing them in a list as the variable `parasites`.

## Calculating few photon coherence functions

We are now ready to calculate low order coherence functions. First let us calculate the `g1` coherence function in the usual units of photonic flux.

```python
import g1

# possible energies of the single incoming photonic state
Es = numpy.linspace(-10 + 1e9, 10 + 1e9, 512)

# the g1 function, where we supply chli as the incoming channel index and chlo as the outgoing channel index
Es, g1res = g1.coherent_state(se, chli=0, chlo=1, Es=Es)
```

The normalized `g2` function (for a coherent state input) can be calculated in the following way:

```python
import g2

# possible energies of the two-photon incoming photonic state
Es = 2 * numpy.linspace(-10 + 1e9, 10 + 1e9, 512)

# the g2 function, where we supply chli as the incoming channel indices and chlo as the outgoing channel indices
g2res = g2.coherent_state(se, chlis=(0, 0), chlso-(1, 1), Es=Es)

# g2res['g2'] now contains g2, but also several components of it for easy access
```
