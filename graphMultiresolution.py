import numpy as np
from pygsp import graphs, reduction

# Reduce a graph using the PyGSP

# Sensor graph with 512 nodes
G = graphs.Sensor(512, distribute=True)
G.compute_fourier_basis()

# The function graph_multiresolution computes the graph pyramid
levels = 5
Gs = reduction.graph_multiresolution(G, levels, sparsify=False)

# Compute the fourier basis of our different graph layers
for gr in Gs:
    gr.compute_fourier_basis()
    
# Let’s now create two signals and a filter, resp f, f2 and g    
f = np.ones((G.N))
f[np.arange(G.N//2)] = -1
f = f + 10*Gs[0].U[:, 7]

f2 = np.ones((G.N, 2))
f2[np.arange(G.N//2)] = -1

g = [lambda x: 5./(5 + x)]

# Run the analysis of the two signals on the pyramid and obtain a coarse approximation 
# for each layer, with decreasing number of nodes. Additionally, we will also get prediction 
# errors at each node at every layer
ca, pe = reduction.pyramid_analysis(Gs, f, h_filters=g, method='exact')
ca2, pe2 = reduction.pyramid_analysis(Gs, f2, h_filters=g, method='exact')

# Given the pyramid, the coarsest approximation and the prediction errors, we will now reconstruct the original signal on the full graph
f_pred, _ = reduction.pyramid_synthesis(Gs, ca[levels], pe, method='exact')
f_pred2, _ = reduction.pyramid_synthesis(Gs, ca2[levels], pe2, method='exact')

# Final errors for each signal after reconstruction
err = np.linalg.norm(f_pred-f)/np.linalg.norm(f)
err2 = np.linalg.norm(f_pred2-f2)/np.linalg.norm(f2)
assert (err < 1e-10) & (err2 < 1e-10)
