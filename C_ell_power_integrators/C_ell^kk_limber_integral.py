import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.scipy.ndimage import map_coordinates
from jax import vmap
import functools
import matplotlib.pyplot as plt
from classy_sz import Class as Class_sz
import pickle
import jax.numpy as jnp
from flax import linen as nn
from scipy.interpolate import interp1d
import os
import time
from math import e

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
    return the interpolated power spectrum as a JAX array. For z > 5, the result is
    adjusted as P(z=5)*(1+5)^2/(1+z)^2.
    """
    # Define grid parameters.
    lower_bound = 1e-5
    upper_bound = 5.0
    num_points = predicted_test_Y.shape[0]  # expected to be 10
    num_lnk = predicted_test_Y.shape[1]       # expected to be 200

    # Create the grid on each axis.
    x = jnp.logspace(jnp.log10(lower_bound), jnp.log10(upper_bound), num=num_points)
    lnk = jnp.logspace(-4, 1, num=num_lnk)

    # Clamp z values to 5 for the purpose of interpolation.
    z_clamped = jnp.minimum(z, 5.0)

    # Compute fractional index along the first axis using the clamped z.
    idx0 = (jnp.log10(z_clamped) - jnp.log10(x[0])) / (jnp.log10(x[-1]) - jnp.log10(x[0])) * (num_points - 1)

    # For axis 1 (lnk), compute the fractional index for the given k (a scalar).
    idx1_scalar = (jnp.log10(k) - (-4.)) / (1 - (-4)) * (num_lnk - 1)
    # Broadcast this scalar index to have the same shape as idx0.
    idx1 = jnp.full_like(idx0, idx1_scalar)

    # Stack the fractional indices into a coordinate array for map_coordinates.
    coords = jnp.stack([idx0, idx1], axis=0)
    interp_flat = map_coordinates(predicted_test_Y, coords, order=1, mode='nearest')
    interp_vals = jnp.reshape(interp_flat, z.shape)

    # Convert from log-space back to physical power values.
    P_at_z = jnp.power(10., interp_vals)

    # For z > 5, adjust the power using the formula: P(z=5)*(1+5)^2/(1+z)^2.
    # Since we already clamped z to 5 in the interpolation, P_at_z is equal to P(z=5) for any z > 5.
    mask = z > 5.0
    scale_factor = ((1 + 5) ** 2) / ((1 + z) ** 2)
    P_adjusted = P_at_z * scale_factor

    # Use jnp.where to choose the adjusted value when z > 5.
    return jnp.where(mask, P_adjusted, P_at_z)

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

def build_nn_parameter_dict(cosmo_params):
    """
    Convert cosmo_params (dictionary) into the dictionary your NN expects.
    For example, if your NN's input vector = (ombh2, omch2, H0, ln10As, ns).
    Adjust names and formulas to match your training.
    """
    return {
       'ombh2':  cosmo_params['omega_b'],      # e.g. was trained with ombh2
       'omch2':  cosmo_params['omega_cdm'],    # e.g. was trained with omch2
       'H0':     cosmo_params['H0'],
       'logA':   cosmo_params['ln10^{10}A_s'],
       'ns':     cosmo_params['n_s'],
    }

def compute_cmb_lensing_cl_emulator_jax_fast(
    cosmo_params,
    z_vals, x_vals,
    lmin=2, lmax=2000
):


    derived  = cs.get_derived_parameters(cosmo_params)
    chi_star = derived["chi_star"]

    D_A      = cs.get_angular_distance_at_z(z_vals, cosmo_params)
    chi_vals = D_A*(1+z_vals)
    dchi_dz  = np.gradient(chi_vals, z_vals)
    a_vals   = 1.0/(1.0+z_vals)


    h        = cosmo_params["H0"]/100.
    Omega_m  = (cosmo_params["omega_b"] + cosmo_params["omega_cdm"])/(h**2)
    prefactor= 1.5*Omega_m*(cosmo_params["H0"]**2)/(c_light**2)

    W_vals   = prefactor*(chi_vals/a_vals)*((chi_star - chi_vals)/chi_star)

    # Convert these geometry arrays to JAX
    z_j   = jnp.array(z_vals)
    chi_j = jnp.array(chi_vals)
    dchi_j= jnp.array(dchi_dz)
    W_j   = jnp.array(W_vals)
    x_j   = jnp.array(x_vals)

    # Precompute the neural network emulator grid once.
    nn_params = build_nn_parameter_dict(cosmo_params)
    predicted_grid = emulate_jax(nn_params)  # precomputed emulator grid

    # Build the multipole array
    ell_array = jnp.arange(lmin, lmax+1)  # shape (n_ell,)
    
    # Reshape geometry arrays for broadcasting:
    # z_array, chi_array, etc. are (n_z,) -> (n_z, 1)
    z_ex    = z_j[:, None]      # shape (n_z, 1)
    chi_ex  = chi_j[:, None]    # shape (n_z, 1)
    dchi_ex = dchi_j[:, None]  # shape (n_z, 1)
    W_ex    = W_j[:, None]      # shape (n_z, 1)
    x_ex    = x_j[:, None]      # shape (n_z, 1)


    # Broadcast ell to shape (1, n_ell)
    ell_ex = ell_array[None, :]      # shape (1, n_ell)

    # Compute k for every combination of redshift and multipole:
    # k = (ell + 0.5) / chi, broadcasting chi_ex and ell_ex.
    k_ex = (ell_ex + 0.5) / chi_ex   # shape (n_z, n_ell)

    # Create a corresponding 2D array for z (each row is the same z value)
    z_here = jnp.broadcast_to(z_ex, k_ex.shape)  # shape (n_z, n_ell)

    # Perform the NN interpolation over the full grid in one go.
    pk_vals = interpolate_jax(predicted_grid, z_here, k_ex / h) / (h ** 3)

    # Compute the multiplicative factor from the geometry.
    factor = (W_ex**2 / chi_ex**2) * dchi_ex * jnp.exp(x_ex)

    # Full integrand on the 2D grid (n_z, n_ell)
    full_integrand = factor * pk_vals

    # Transpose to shape (n_ell, n_z) so integration is over the redshift axis.
    full_integrand_T = jnp.transpose(full_integrand)

    # Integrate using Simpson’s rule over x (with x_array corresponding to log(1+z))
    Cl_kappa = simpson_jax(full_integrand_T, x_j)
    
    return ell_array, Cl_kappa


# ----------------------------------------------------------------------------------------
# 4) A “classic” lensing integrator using cs.get_pknl_at_z(...) [not JAX-accelerated]
# ----------------------------------------------------------------------------------------
def compute_cmb_lensing_cl_classic(cosmo_params, lmin=2, lmax=2000, n_x=200):
    """
    Uses the same Limber logic, but calls CLASS-SZ for P(k,z).
    This cannot be JIT-compiled, as it calls C++ code under the hood.
    """
    derived  = cs.get_derived_parameters(cosmo_params)
    chi_star = derived["chi_star"]

    h        = cosmo_params["H0"]/100.
    Omega_m  = (cosmo_params["omega_b"] + cosmo_params["omega_cdm"])/(h**2)

    z_min, z_max = 1e-4, 20.0
    x_vals       = np.linspace(np.log(1+z_min), np.log(1+z_max), n_x)
    z_vals       = np.exp(x_vals) - 1

    D_A     = cs.get_angular_distance_at_z(z_vals, cosmo_params)
    chi_vals= D_A*(1+z_vals)
    dchi_dz = np.gradient(chi_vals, z_vals)
    a_vals  = 1.0/(1.0+z_vals)

    c_light   = 299792.458
    prefactor = 1.5*Omega_m*(cosmo_params["H0"]**2)/(c_light**2)
    W_vals    = prefactor*(chi_vals/a_vals)*((chi_star-chi_vals)/chi_star)

    ls = np.arange(lmin, lmax+1)
    n_ell = len(ls)
    integrand = np.zeros((n_x, n_ell))

    for i in range(n_x):
        z_here = z_vals[i]
        pks, ks= cs.get_pknl_at_z(z_here, cosmo_params)
        valid  = (ks>0) & (pks>0)
        log_ks = np.log(ks[valid])
        log_pks= np.log(pks[valid])
        interp = interp1d(log_ks, log_pks, kind='linear', fill_value='extrapolate')

        chi_i  = chi_vals[i]
        k_vals = (ls+0.5)/chi_i

        pk_now = np.exp(interp(np.log(k_vals)))
        integrand[i,:] = (W_vals[i]**2/chi_i**2)*pk_now*dchi_dz[i]

    integrand *= np.exp(x_vals)[:,None]
    Cl = np.trapz(integrand, x_vals, axis=0)
    return ls, Cl

# ----------------------------------------------------------------------------------------
# 5) Set up CLASS-SZ and define the cosmology
# ----------------------------------------------------------------------------------------
cs = Class_sz()
cs.initialize_classy_szfast()

cosmo_params = {
    "omega_b":      0.022,   # Omega_b * h^2
    "omega_cdm":    0.12,    # Omega_cdm * h^2
    "H0":           67.5,    # in km/s/Mpc
    "tau_reio":     0.06,
    "ln10^{10}A_s": 3.0,
    "n_s":          0.965,
}

# ----------------------------------------------------------------------------------------
# 6) Precompute geometry in NumPy for the JAX path
# ----------------------------------------------------------------------------------------
#derived  = cs.get_derived_parameters(cosmo_params)
#chi_star = derived["chi_star"]

n_x      = 50  # reduce for speed
z_min, z_max = 1e-4, 20.0
x_vals   = np.linspace(np.log(1+z_min), np.log(1+z_max), n_x)
z_vals   = np.exp(x_vals) - 1.0

c_light  = 299792.458


# ----------------------------------------------------------------------------------------
# 7) Compare:
#    (A) Built-in CLASS lensing  (φφ -> κκ)
#    (B) Our "classic" integrator with CLASS P(k,z)
#    (C) JAX integrator with the NN emulator
# ----------------------------------------------------------------------------------------


# (A) Built-in lensing
cls      = cs.get_cmb_cls(cosmo_params)
ell_all  = cls['ell']
cl_phi   = cls['pp']
cl_kappa = ((ell_all*(ell_all+1))/2.0)**2 * cl_phi


# (B) Our "classic" integrator
ls_classic, Cl_classic = compute_cmb_lensing_cl_classic(
    cosmo_params,
    lmax=2000,
    n_x=200
)

t=[]

for i in range(5000):
    start_time = time.perf_counter()
    ell_jax, Cl_jax = compute_cmb_lensing_cl_emulator_jax_fast(
        cosmo_params,
        z_vals, x_vals,
        lmin=2, lmax=2000
    )
    end_time = time.perf_counter()
    t.append(end_time - start_time)

print(f"Average time: {np.mean(t)}+-{np.std(t)} seconds")


# (C) JAX integrator with the neural‐network
ell_jax, Cl_jax = compute_cmb_lensing_cl_emulator_jax_fast(
    cosmo_params,
    z_vals, x_vals,
    lmin=2, lmax=2000
)


# Convert from JAX arrays to NumPy for plotting
ell_jax_np = np.array(ell_jax)
Cl_jax_np  = np.array(Cl_jax)

# Assume y1 and y2 are your data arrays
r = np.corrcoef(Cl_jax_np, Cl_classic)[0, 1]
r_squared = r ** 2
print("r^2 between CLASS P and NN emulator =", r_squared)

# ----------------------------------------------------------------------------------------
# 8) Plot all three on one log-log scale
# ----------------------------------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.loglog(ell_all, cl_kappa,           label="(1) CLASS-SZ built-in (φφ->κκ)")
plt.loglog(ls_classic, Cl_classic,      label="(2) JAX integrator + CLASS P(k,z)")
plt.loglog(ell_jax_np, Cl_jax_np,       label="(3) JAX integrator + NN emulator")

plt.xlabel(r'Multipole $\ell$')
plt.ylabel(r'$C_\ell^{\kappa \kappa}$')
plt.title("Compare: CLASS vs. Classic Integrator vs. JAX+NN Integrator")
plt.grid(True)
plt.legend()
plt.savefig("C_ell^kk_Plot_First_Arc.png", bbox_inches='tight')
plt.show()