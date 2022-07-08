import numpy as np
from pygsp import graphs, plotting
import pyunlocbox

# Plot of the original label signal, that we wish to recover, on the graph.

# Create a random sensor graph
G = graphs.Sensor(N=256, distribute=True, seed=42)
G.compute_fourier_basis()

# Create label signal
label_signal = np.copysign(np.ones(G.N), G.U[:, 3])

G.plot_signal(label_signal)

# Label signal on the graph after the application of the subsampling mask and the addition 
# of noise.
rs = np.random.RandomState(42)

# Create the mask
M = rs.rand(G.N)
M = (M > 0.6).astype(float)  # Probability of having no label on a vertex.

# Applying the mask to the data
sigma = 0.1
subsampled_noisy_label_signal = M * (label_signal + sigma * rs.standard_normal(G.N))

G.plot_signal(subsampled_noisy_label_signal)

# Here we solve the optimization problem
# Set the functions in the problem
gamma = 3.0
d = pyunlocbox.functions.dummy()
r = pyunlocbox.functions.norm_l1()
f = pyunlocbox.functions.norm_l2(w=M, y=subsampled_noisy_label_signal,lambda_=gamma)

# Define the solver
G.compute_differential_operator()
L = G.D.toarray()
step = 0.999 / (1 + np.linalg.norm(L))
solver = pyunlocbox.solvers.mlfbf(L=L, step=step)

# Solve the problem
x0 = subsampled_noisy_label_signal.copy()
prob1 = pyunlocbox.solvers.solve([d, r, f], solver=solver, x0=x0, rtol=0, maxit=1000)

G.plot_signal(prob1['sol'])

# Set the functions in the problem
r = pyunlocbox.functions.norm_l2(A=L, tight=False)
# Define the solver
step = 0.999 / np.linalg.norm(np.dot(L.T, L) + gamma * np.diag(M), 2)
solver = pyunlocbox.solvers.gradient_descent(step=step)

# Solve the problem
x0 = subsampled_noisy_label_signal.copy()
prob2 = pyunlocbox.solvers.solve([r, f], solver=solver,x0=x0, rtol=0, maxit=1000)

G.plot_signal(prob2['sol'])