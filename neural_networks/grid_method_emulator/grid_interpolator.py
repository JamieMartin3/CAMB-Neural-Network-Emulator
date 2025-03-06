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
import time
import random
from scipy.interpolate import RegularGridInterpolator

######################################################## Load Testing Data ########################################################

shared_drive_path = '/home/jam249/rds/dlproject/'

test_X = np.load(shared_drive_path + 'logs/z_method_params_data_999.npy',allow_pickle=True).item()
test_Y_unlog = np.load(shared_drive_path + 'logs/z_method_pk_data_999.npy', allow_pickle=True)
test_Y = np.log10(test_Y_unlog.astype(np.float32))

# Distributions for z and k, both logarithmically defined
lower_bound = 1e-5
upper_bound = 5
num_points = 10

x = np.logspace(np.log10(lower_bound), np.log10(upper_bound), num=num_points)
lnk = np.logspace(-4,1,200)

# Dictionaries for z and universe parameters
z = test_X[list(test_X.keys())[5]]
test_X = {k: v for k, v in list(test_X.items())[:5]}


# Convert to pandas DataFrame for consistency with your original code
pdict = pd.DataFrame(test_X)
z = pd.DataFrame(z)
test_Y = pd.DataFrame(test_Y)

# Number of trials
num_iterations = 100

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
with open(shared_drive_path + 'grid_emulator.pkl', 'rb') as f:
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

######################################################## Emulator Predictions ########################################################

predicted_test_Y = []
run_time = []

# Loop over each test sample.
for i in range(num_iterations):
    # Construct a dictionary with a single test sample per feature.
    current_pdict = {key: [pdict[key][i]] for key in pdict.keys()}
    
    start_time = time.perf_counter()
    # Get prediction in log10 space.
    prediction = predictions_flax(current_pdict)
    end_time = time.perf_counter()
    
    predicted_test_Y.append(prediction)
    run_time.append(end_time - start_time)

# Compute average and standard deviation of runtime per prediction.
avg_run_time = np.mean(run_time)
sd_run_time = np.std(run_time)

# Convert predictions list to a NumPy array.
predicted_test_Y = np.array(predicted_test_Y)

# Remove the extra dimension if predictions_flax returns a singleton batch dimension.
predicted_test_Y = np.squeeze(predicted_test_Y, axis=1)

# Reshape the predictions to (100, 10, 200) as desired.
predicted_test_Y = predicted_test_Y.reshape(100, 10, 200)

print(f"Average run time: {avg_run_time} ± {sd_run_time} seconds")

######################################################## Interpolation ########################################################

# --- Compute the percentage differences ---
percent_diff = []

for i in range(num_iterations):
    # Select a random column index from the second grid (0 to 199)
    rand = random.randint(0, 199)
    
    # Create an interpolator for the i-th test sample's predicted 2D grid.
    # predicted_test_Y[i] should have shape (10, 200).
    interpolator = RegularGridInterpolator((x, lnk), predicted_test_Y[i],
                                             method="cubic", bounds_error=True)
    
    # Create the test point:
    #   - The first coordinate is taken from z (for the i-th test sample).
    #   - The second coordinate is the value from lnk at the randomly chosen index.
    test_point = [z.iloc[i].item(), lnk[rand]]
    
    # Get the interpolated value at the test point.
    interp_value = interpolator(test_point)
    
    # Get the corresponding true value from test_Y.
    true_value = test_Y.iloc[i, rand]
    
    # Compute the relative difference (absolute value) and store it.
    diff = np.abs((interp_value - true_value) / true_value)
    percent_diff.append(diff)

# Compute the mean and standard deviation (in percent).
mean_perc_diff = np.mean(percent_diff) * 100
sd_perc_diff = np.std(percent_diff) * 100

print(f"The mean percentage difference between the interpolated value and true value is {mean_perc_diff} ± {sd_perc_diff}")
