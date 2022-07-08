import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting, utils

G = graphs.Bunny()

# Filters with different heat kernel sizes
taus = [10, 25, 50]
g = filters.Heat(G, taus)

# Signal on one vertex, heat source
s = np.zeros(G.N)
DELTA = 2000
s[DELTA] = 1

# Simulating heat diffusion
s = g.filter(s, method='chebyshev')

# Plot the heat diffusion for the different heat kernels
fig = plt.figure(figsize=(10, 3))

for i in range(g.Nf):
    
    ax = fig.add_subplot(1, g.Nf, i+1, projection='3d')
    G.plot_signal(s[:, i], colorbar=False, ax=ax)
    title = r'Heat diffusion, $\tau={}$'.format(taus[i])
    _ = ax.set_title(title)
    ax.set_axis_off()

fig.tight_layout()

# Filter bank of wavelets.
g = filters.MexicanHat(G, Nf=6)  # Nf = 6 filters in the filter bank.

# Plot the filter bank of wavelets
fig, ax = plt.subplots(figsize=(10, 5))
g.plot(ax=ax)
_ = ax.set_title('Filter bank of mexican hat wavelets')


# Filtering a Kronecker delta placed at one specific vertex.
s = g.localize(DELTA)
fig = plt.figure(figsize=(10, 2.5))

for i in range(3):

    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    G.plot_signal(s[:, i], ax=ax)
    _ = ax.set_title('Wavelet {}'.format(i+1))
    ax.set_axis_off()

fig.tight_layout()

s = G.coords
s = g.filter(s)

s = np.linalg.norm(s, ord=2, axis=1)

fig = plt.figure(figsize=(10, 7))
for i in range(4):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    G.plot_signal(s[:, i], ax=ax)
    title = 'Curvature estimation (scale {})'.format(i+1)
    _ = ax.set_title(title)
    ax.set_axis_off()
fig.tight_layout()