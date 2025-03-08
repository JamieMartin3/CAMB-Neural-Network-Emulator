import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.scipy.ndimage import map_coordinates
from jax import vmap
# from jax.scipy.integrate import trapezoid
import functools
import matplotlib.pyplot as plt
from classy_sz import Class as Class_sz
# from scipy.integrate import simpson
import pickle
import jax.numpy as jnp
from flax import linen as nn
from scipy.stats import norm
import os
import time

################################################################### Initialisation ###################################################################

allpars = {
    'H0': 70.,
    'ombh2': 0.0225,
    'omch2':  0.12,
    'logA': 3.047,
    'ns': 0.9665,
}

# Initialize Class_sz and set parameters using the correct dictionary
classy_sz = Class_sz()
classy_sz.set(allpars)
classy_sz.compute_class_szfast()

################################################################### Emulating and Interpolating Power Spectrum ###################################################################

#-----------------------------------------------------------
# Re-define the custom activation and network modules
#-----------------------------------------------------------
def custom_activation(x, alpha, beta):
    """
    Custom activation function:
      f(x) = x * [ beta + sigmoid(alpha * x) * (1 - beta) ]
    """
    return x * (beta + jax.nn.sigmoid(alpha * x) * (1 - beta))

class CustomDense(nn.Module):
    """A Dense layer that applies a custom activation if desired."""
    features: int
    use_activation: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.features,
            kernel_init=nn.initializers.normal(1e-3),
            bias_init=nn.initializers.zeros
        )(x)
        if self.use_activation:
            # Define trainable parameters for the custom activation.
            alpha = self.param('alpha', nn.initializers.normal(), (self.features,))
            beta  = self.param('beta',  nn.initializers.normal(), (self.features,))
            x = custom_activation(x, alpha, beta)
        return x

class Emulator(nn.Module):
    """The overall network architecture."""
    hidden_layers: list  # list of neurons per hidden layer
    output_dim: int

    @nn.compact
    def __call__(self, x):
        for features in self.hidden_layers:
            x = CustomDense(features, use_activation=True)(x)
        # Final linear layer without activation
        x = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.normal(1e-3),
            bias_init=nn.initializers.zeros
        )(x)
        return x

#-----------------------------------------------------------
# Load the saved model attributes
#-----------------------------------------------------------
# Adjust the file path as necessary.
with open('/home/jam249/rds/dlproject/grid_emulator.pkl', 'rb') as f:
    attributes = pickle.load(f)

# Retrieve saved parameters and attributes.
params   = attributes['params']
X_mean   = attributes['X_mean']
X_std    = attributes['X_std']
Y_mean   = attributes['Y_mean']
Y_std    = attributes['Y_std']
dim_X    = attributes['dim_X']
names_X  = attributes['names_X'] 
n_hidden = attributes['n_hidden']
n_modes  = attributes['n_modes']
modes    = attributes['modes']

#-----------------------------------------------------------
# Re-instantiate the model with the saved architecture.
#-----------------------------------------------------------
model = Emulator(hidden_layers=n_hidden, output_dim=n_modes)

#-----------------------------------------------------------
# Define helper functions to apply the model.
#-----------------------------------------------------------
def apply_model(params, x):
    """
    Standardizes input, runs the model, then un-standardizes the output.
    Expects x to be a JAX array.
    """
    # Standardize the input using training statistics.
    x_std = (x - jnp.array(X_mean)) / jnp.array(X_std)
    # Run the model.
    y = model.apply(params, x_std)
    # Un-standardize the output.
    return y * jnp.array(Y_std) + jnp.array(Y_mean)


# Define a helper function to perform predictions using the loaded model.
def predictions_flax(parameters_dict):
    """
    Given a dict of input parameters (with keys matching names_X),
    returns model predictions using the loaded Flax parameters.
    """
    # Convert the dict-of-lists into an array of shape (N_samples, dim_X)
    parameters_arr = jnp.stack([jnp.atleast_1d(parameters_dict[k]) for k in names_X], axis=1)
    # Use your previously defined apply_model function (which standardizes inputs and un-standardizes outputs)
    return apply_model(params, parameters_arr)

###############################################################################
# Define functions to emulate and interpolate the power spectrum
###############################################################################

def emulate_jax(params):
    """
    Given a dict of input parameters (keys must match names_X),
    returns model predictions using the loaded Flax parameters,
    as a JAX array of shape (10, 200).
    """
    # Call the prediction function (assumed to be defined and to work with JAX arrays)
    predicted_test = predictions_flax(params)
    # Convert the output to a JAX array.
    predicted_test = jnp.asarray(predicted_test)
    # Squeeze the first axis if it is a singleton.
    # (We assume the shape is known at compile time; otherwise, use lax.cond.)
    if predicted_test.shape[0] == 1:
        predicted_test = jnp.squeeze(predicted_test, axis=0)
    # Reshape into the desired grid shape.
    predicted_test = jnp.reshape(predicted_test, (10, 200))
    return predicted_test


def interpolate_jax(predicted_test_Y, z, k):
    """
    Given a grid of emulated power spectrum values (predicted_test_Y) defined
    on a regular grid with coordinates (x, lnk) and given query coordinates z and k,
    return the interpolated power spectrum as a JAX array.
    
    The original (NumPy) version creates:
      x = logspace(log10(lower_bound), log10(upper_bound), num=num_points)
      lnk = logspace(-4, 1, 200)
    Here we do the same but then convert physical query coordinates into fractional indices.
    
    We assume:
      - predicted_test_Y is a JAX array of shape (num_points, 200)  (here, (10,200)).
      - For the first grid axis, the physical coordinate is defined on x.
      - For the second grid axis, the physical coordinate is defined on lnk.
    Finally, we return 10 ** (interpolated value) to match the original behavior.
    """
    # Define grid parameters.
    lower_bound = 1e-5
    upper_bound = 5.0
    num_points = predicted_test_Y.shape[0]  # expected to be 10
    num_lnk = predicted_test_Y.shape[1]       # expected to be 200

    # Create the grid on each axis (as JAX arrays).
    # Note: Although these are computed via jnp.logspace, they are computed once.
    x = jnp.logspace(jnp.log10(lower_bound), jnp.log10(upper_bound), num=num_points)
    # For the second axis, note that the original code used np.logspace(-4, 1, 200).
    lnk = jnp.logspace(-4, 1, num=num_lnk)

    # Convert query coordinates (z and k) to fractional indices.
    # For axis 0 (x): we assume the physical coordinate is z.
    # Compute: index = (log10(z) - log10(x[0])) / (log10(x[-1]) - log10(x[0])) * (num_points-1)
    idx0 = (jnp.log10(z) - jnp.log10(x[0])) / (jnp.log10(x[-1]) - jnp.log10(x[0])) * (num_points - 1)
    # For axis 1 (lnk): since the grid is defined in physical k, but the grid values are log-spaced,
    # we convert the query k into an index using log10.
    idx1 = (jnp.log10(k) - (-4.)) / (1 - (-4)) * (num_lnk - 1)

    # Stack the fractional indices into a coordinate array for map_coordinates.
    # map_coordinates expects coordinates with shape (ndim, n_points).
    coords = jnp.stack([idx0, idx1], axis=0)
    # If z and k are arrays of shape (N,), then coords has shape (2, N).

    # Use map_coordinates with order=3 for cubic interpolation.
    # This performs interpolation on the predicted_test_Y array.
    interp_flat = map_coordinates(predicted_test_Y, coords, order=1, mode='nearest')
    # Reshape the result to match the shape of the input queries.
    interp_vals = jnp.reshape(interp_flat, z.shape)

    # Convert from log-space back to physical values.
    return jnp.power(10., interp_vals)

################################################################### Simpson Integration Function ###################################################################

def simpson_jax(y, x, axis=-1):
    """
    Approximate the integral of y along the specified axis using Simpson's rule,
    given sample positions x (assumed to be uniformly spaced).

    Parameters:
      y    : array-like of function values.
      x    : array-like integration grid (for example, z_vals).
      axis : axis along which to integrate.

    Returns:
      The approximate integral.
    """
    y = jnp.asarray(y)
    x = jnp.asarray(x)
    N = y.shape[axis]
    
    # Compute the uniform spacing from the grid.
    dx = x[1] - x[0]
    
    # Check if the number of samples is odd.
    if N % 2 == 1:
        # Use Simpson's rule over the whole interval.
        indices = jnp.arange(N)
        coeff = jnp.where(
            (indices == 0) | (indices == N - 1),
            1.0,
            jnp.where(indices % 2 == 1, 4.0, 2.0)
        )
        return dx / 3 * jnp.sum(coeff * y, axis=axis)
    else:
        # For even number of samples, apply Simpson's rule to the first N-1 points
        # and the trapezoidal rule to the last interval.
        n = N - 1
        indices = jnp.arange(n)
        coeff = jnp.where(
            (indices == 0) | (indices == n - 1),
            1.0,
            jnp.where(indices % 2 == 1, 4.0, 2.0)
        )
        simpson_part = dx / 3 * jnp.sum(coeff * jnp.take(y, indices, axis=axis), axis=axis)
        trapezoidal_part = dx * 0.5 * (jnp.take(y, N - 2, axis=axis) + jnp.take(y, N - 1, axis=axis))
        return simpson_part + trapezoidal_part

################################################################### C_ell Plotting ###################################################################

# -------------------------------------------------------------------
# JAX-compatible weight function
# -------------------------------------------------------------------
def weight_function_jax(z):
    return jnp.exp(-0.5 * ((z - 3.)/0.1)**2) / (0.1 * jnp.sqrt(2*jnp.pi))

# -------------------------------------------------------------------
# JAX-compatible cl_limber function using the new interpolate_jax and emulate_jax
# -------------------------------------------------------------------
def cl_limber_jax(l, allpars, zmin=1., zmax=20., nz=50, predicted_grid=None):

    # Create a redshift grid.
    z_vals = jnp.logspace(0, jnp.log10(zmax), nz)
    
    # Convert outputs of non-JAX cosmology functions to JAX arrays.
    D_A_vals = jnp.asarray(classy_sz.get_angular_distance_at_z(z_vals, params_values_dict=allpars)) 
    H_vals   = jnp.asarray(classy_sz.get_hubble_at_z(z_vals, params_values_dict=allpars)) 
    
    # Compute comoving distance: chi = D_A * (1 + z)
    chi_vals = D_A_vals * (1. + z_vals)
    
    # Evaluate the weight function.
    W_vals = 3 / 2 * (allpars['ombh2'] + allpars['omch2']) * 1e4 * chi_vals * (1 + z_vals) * (chi_vals[-1] - chi_vals) / chi_vals[-1]
    
    # Compute k for every z.
    k_vals = (l + 0.5) / chi_vals
    
    # Use our JAX-friendly interpolation.
    Pk_vals = interpolate_jax(predicted_grid, z_vals, k_vals)
    
    # Compute the integrand.
    integrand = (H_vals / chi_vals**2) * (W_vals**2) * Pk_vals
    


    # Integrate using the JAX-friendly trapz.
    return simpson_jax(integrand, z_vals)

# -------------------------------------------------------------------
# Main execution: emulate, vectorize cl_limber, and plot.
# -------------------------------------------------------------------

# Convert predicted emulator grid using our JAX-friendly emulate function.
predicted_grid_em = emulate_jax(allpars)

# Create a range of multipoles as a JAX array.
l_vals_jax = jnp.logspace(jnp.log10(2.), jnp.log10(10000.), 100)

start_time = time.perf_counter()

# Use vmap to vectorize cl_limber_jax over l.
vectorized_cl = vmap(lambda l: cl_limber_jax(l, allpars, predicted_grid=predicted_grid_em))
Cl_vals_jax = vectorized_cl(l_vals_jax)

end_time = time.perf_counter()
print("Run time:", end_time - start_time, "seconds")

# Convert JAX arrays to NumPy arrays for plotting.
l_vals_np = jnp.asarray(l_vals_jax)
Cl_vals_np = jnp.asarray(Cl_vals_jax)

plt.figure(figsize=(8,5))
plt.plot(l_vals_np, Cl_vals_np, label=r'$C_\ell$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^k$')
plt.title(r'$C_\ell^k$ via Limber Approximation')
plt.legend()
plt.tight_layout()
plt.savefig('/home/jam249/rds/dlproject/C_ell^kk_Plot')