from pygsp import graphs, filters
from pygsp import plotting
import matplotlib.pylab as plt
import networkx as nx
import numpy as np

# Graph of minessota road map
G = graphs.Minnesota()
G_nx = nx.Graph([tuple(r) for r in np.array(G.get_edge_list()[:2]).T])

# Plot the graph
#plotting.plot_graph(G)

# Noisy signal based on the distance from the dense part of Minnesota
rs = np.random.RandomState()
s = np.zeros(G.N)
s += np.sqrt(np.sum((G.coords - np.array([-93.2, 45]))**2, axis=1))
s[s>2] = 3
s += rs.uniform(-1,1, size=G.N)

# Plot the graph with the signal
plotting.plot_signal(G, s)

# First design LPF filter as a heat kernel
g = filters.Heat(G, tau=50)

# Here we filter the signal s in the graph G
s_out = g.filter(s, method='exact') # exact uses GFT. Chebychev approx. is also available

# Plotting code
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
plotting.plot_signal(G, s, ax=axes[0])
_ = axes[0].set_title('Noisy signal before filtering',fontsize=15)
axes[0].set_axis_off()
plotting.plot_signal(G, s_out, ax=axes[1])
_ = axes[1].set_title('Filtered signal',fontsize=15)
axes[1].set_axis_off()
fig.tight_layout()
plt.show()

