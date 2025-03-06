from typing_extensions import final
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from functools import partial
from IPython.display import clear_output

######################################################## Load Testing Data ########################################################


shared_drive_path = '/home/jam249/rds/dlproject/'

test_X = np.load(shared_drive_path + 'logs/z_method_params_combined_20.npy',allow_pickle=True).item()
test_Y_unlog = np.load(shared_drive_path + 'logs/z_method_pk_combined_20.npy', allow_pickle=True)
reshaped_array = test_Y_unlog.reshape(-1, 200)
test_Y = np.log10(reshaped_array.astype(np.float32))

# Convert to pandas DataFrame for consistency with your original code
test_X = pd.DataFrame(test_X)
test_Y = pd.DataFrame(test_Y)

# Convert to NumPy arrays of type float32
X_data_test = test_X.values.astype(np.float32)
Y_data_test = test_Y.values.astype(np.float32)

######################################################## Load Emulator ########################################################

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
with open(shared_drive_path + 'z_emulator.pkl', 'rb') as f:
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
# Define a helper function to apply the model.
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
def predictions_flax(pdict_test_X):
    """
    Given a dictionary of test inputs (with keys corresponding to the training
    feature names), order the data properly, convert to a JAX array, and run the model.
    """
    # Create a NumPy array with columns ordered as in names_X.
    # We assume that pdict_test_X is a dictionary with lists as values.
    test_X_ordered = np.column_stack([pdict_test_X[key] for key in names_X])
    # Convert to JAX array.
    test_X_jax = jnp.array(test_X_ordered)
    # Run the model.
    preds = apply_model(params, test_X_jax)
    return np.array(preds)

######################################################## Testing ########################################################

# Convert test parameters (assumed to be in a DataFrame) to a dict.
pdict_test_X = test_X.to_dict(orient='list')

# Run predictions (predictions will be in log10 space as before)
predicted_test_Y = predictions_flax(pdict_test_X)
predicted_test_Y_np = np.array(predicted_test_Y)

# Convert both predictions and true test values from log10 to linear space.
true_test_Y_linear = 10 ** test_Y.values
pred_test_Y_linear = 10 ** predicted_test_Y_np

# Compute the relative difference (in percent).
diff = 100. * np.abs(pred_test_Y_linear - true_test_Y_linear) / true_test_Y_linear

# Compute percentiles for each mode across the test samples.
percentiles = np.zeros((4, diff.shape[1]))
percentiles[0] = np.percentile(diff, 68, axis=0)
percentiles[1] = np.percentile(diff, 95, axis=0)
percentiles[2] = np.percentile(diff, 99, axis=0)
percentiles[3] = np.percentile(diff, 99.9, axis=0)

# Plot the percentile bands.
plt.figure(figsize=(10, 5))
plt.fill_between(modes, 0, percentiles[2, :], color='salmon',  label='99%', alpha=0.8)
plt.fill_between(modes, 0, percentiles[1, :], color='red',     label='95%', alpha=0.7)
plt.fill_between(modes, 0, percentiles[0, :], color='darkred', label='68%', alpha=1.0)
plt.legend(frameon=False, fontsize=10, loc=1)
plt.ylabel(r'$\frac{|P_{\rm NN} - P_{\rm true}|}{P_{\rm true}} \times 100\%$', fontsize=15)
plt.xlabel(r'$k$', fontsize=13)
plt.xscale('log')
plt.grid(which='both', alpha=0.4)
plt.tight_layout()

save_path = 'z_percentile_plot.png'
plt.savefig(shared_drive_path + save_path)
print(f"Plot saved to {save_path}")
