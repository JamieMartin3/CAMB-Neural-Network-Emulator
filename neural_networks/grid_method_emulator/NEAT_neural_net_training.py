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
import wandb
import neat
wandb.login()

#-----------------------------------------------------------
# Data loading and pre-processing
#-----------------------------------------------------------
# Load training data (adjust paths as needed)

train_X = np.load('/home/jam249/rds/dlproject/grid_method_params_combined_10.npy',allow_pickle=True).item()
train_Y_unlog = np.load('/home/jam249/rds/dlproject/grid_method_pk_combined_10.npy', allow_pickle=True)
train_Y = np.log10(train_Y_unlog.astype(np.float32))
train_Y = train_Y.reshape(-1, 2000)

# Convert to pandas DataFrame for consistency with your original code
train_X = pd.DataFrame(train_X)
train_Y = pd.DataFrame(train_Y)

# Convert to NumPy arrays of type float32
X_data = train_X.values.astype(np.float32)
Y_data = train_Y.values.astype(np.float32)

# Compute feature and label statistics for standardisation
X_mean = np.mean(X_data, axis=0, keepdims=True)
X_std  = np.std(X_data, axis=0, keepdims=True)
Y_mean = np.mean(Y_data, axis=0, keepdims=True)
Y_std  = np.std(Y_data, axis=0, keepdims=True)

# Convert these constants to JAX arrays
X_mean_jax = jnp.array(X_mean)
X_std_jax  = jnp.array(X_std)
Y_mean_jax = jnp.array(Y_mean)
Y_std_jax  = jnp.array(Y_std)

#-----------------------------------------------------------
# Training schedule parameters
#-----------------------------------------------------------
validation_split = 0.1
N_total = X_data.shape[0]
n_validation = int(N_total * validation_split)
n_training   = N_total - n_validation

# Create a training/validation split.
mask = np.array([True] * n_training + [False] * n_validation)
np.random.shuffle(mask)
X_train = X_data[mask]
Y_train = Y_data[mask]
X_val   = X_data[~mask]
Y_val   = Y_data[~mask]

# Additional architecture attributes (as in your original code)
names_X = list(train_X.keys())
dim_X   = len(names_X)
modes   = np.logspace(-4, 46, 2000)   # grid points (k's)
n_modes = len(modes)

# =============================================================================
# Model Definitions (Custom activation, Dense layer, and Emulator)
# =============================================================================

def custom_activation(x, alpha, beta):
    """Custom activation function: f(x) = x * [ beta + sigmoid(alpha * x) * (1 - beta) ]"""
    return x * (beta + jax.nn.sigmoid(alpha * x) * (1 - beta))

class FlexibleDense(nn.Module):
    features: int
    activation: str = "custom"  # Options: "custom", "relu", "tanh", "sigmoid"

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.features,
            kernel_init=nn.initializers.normal(1e-3),
            bias_init=nn.initializers.zeros
        )(x)
        if self.activation == "custom":
            alpha = self.param('alpha', nn.initializers.normal(), (self.features,))
            beta  = self.param('beta', nn.initializers.normal(), (self.features,))
            x = custom_activation(x, alpha, beta)
        elif self.activation == "relu":
            x = jax.nn.relu(x)
        elif self.activation == "tanh":
            x = jax.nn.tanh(x)
        elif self.activation == "sigmoid":
            x = jax.nn.sigmoid(x)
        return x

class Emulator(nn.Module):
    hidden_layers: list  # list of integers (neurons per layer)
    output_dim: int
    activation: str = "custom"  # Which activation to use in each layer.

    @nn.compact
    def __call__(self, x):
        for features in self.hidden_layers:
            x = FlexibleDense(features, activation=self.activation)(x)
        x = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.normal(1e-3),
            bias_init=nn.initializers.zeros
        )(x)
        return x

# =============================================================================
# Helper Functions: Apply model, Batch generator, Save model
# =============================================================================

def apply_model(params, x, model):
    """Standardize input, run through the network, and un-standardize the output."""
    x_std = (x - X_mean_jax) / X_std_jax
    y = model.apply(params, x_std)
    return y * Y_std_jax + Y_mean_jax

def get_batches(X, Y, batch_size):
    """Yield mini-batches (shuffled each epoch)."""
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], Y[batch_idx]

def save_model(params, filename):
    """Save the model parameters and additional attributes."""
    params_cpu = jax.device_get(params)
    attributes = {
        'params': params_cpu,
        'X_mean': np.array(X_mean),
        'X_std':  np.array(X_std),
        'Y_mean': np.array(Y_mean),
        'Y_std':  np.array(Y_std),
        'dim_X':  dim_X,
        'names_X': names_X,
        'n_modes': n_modes,
        'modes': modes,
        'architecture': [dim_X] + n_hidden + [n_modes]  # n_hidden must be defined in context
    }
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(attributes, f)
    print(f"Model saved to {filename}.pkl")

# =============================================================================
# JIT-Compiled Training and Evaluation Functions (using JAX)
# =============================================================================

@partial(jax.jit, static_argnums=(0,5))
def train_step(optimizer, params, opt_state, x, y, model_config):
    # Unpack model configuration.
    n_hidden, output_dim, activation = model_config
    # Recreate the model instance from configuration.
    model_instance = Emulator(hidden_layers=n_hidden, output_dim=output_dim, activation=activation)

    def loss_fn(params):
        preds = apply_model(params, x, model_instance)
        return jnp.sqrt(jnp.mean((preds - y) ** 2))

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@partial(jax.jit, static_argnums=(3,))
def compute_loss(params, x, y, model_config):
    n_hidden, output_dim, activation = model_config
    model_instance = Emulator(hidden_layers=n_hidden, output_dim=output_dim, activation=activation)
    preds = apply_model(params, x, model_instance)
    loss = jnp.sqrt(jnp.mean((preds - y) ** 2))
    return loss


# =============================================================================
# NEAT Evaluation Functions
# =============================================================================
#
# In this example we use a NEAT genome (via neat-python) to encode the hyperparameters.
# We assume that the genome’s network has 13 output nodes (when activated on a dummy input)
# which are decoded (roughly) as:
#
#  0: activation (0-1 mapped to one of ["custom", "relu", "tanh", "sigmoid"])
#  1: init_lr   (mapped on a log scale from 1e-1 to 1)
#  2: final_lr  (mapped on a log scale from 1e-5 to 1e-2)
#  3: warmup_epochs (mapped to an integer in [50, 100])
#  4: decay_epochs  (mapped to an integer in [50, 200])
#  5: n_layers      (mapped to an integer in [3, 6])
#  6-11: n_units for up to 6 layers (each mapped to [128, 1024] and then quantized to multiples of 128)
# 12: batch_size (mapped to an integer in [1000, 10000] in steps of 1000)
#
# The training loop is then run for the given hyperparameters and the best validation loss is recorded.
# Since NEAT maximizes fitness, we define fitness as the negative of the validation loss.

# Global variable to track the current generation.
current_generation = 0

# -----------------------------------------------------------------------------
# Custom Reporter to Update Generation Number in WandB Runs
# -----------------------------------------------------------------------------
class WandbReporter(neat.reporting.BaseReporter):
    def __init__(self):
        self.generation_counter = 0

    def post_evaluate(self, config, population, species, best_genome):
        # Increment our counter for each generation.
        self.generation_counter += 1
        global current_generation
        current_generation = self.generation_counter

# =============================================================================
# NEAT Evaluation Functions
# =============================================================================

def eval_genome(genome, config):
    # Create a feed-forward network from the genome.
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # Use a dummy input (the network is used solely for hyperparameter generation).
    dummy_input = [1.0]
    output = net.activate(dummy_input)
    # Ensure we have at least 13 outputs (pad if necessary)
    if len(output) < 13:
        output = output + [0]*(13 - len(output))

    # Decode the hyperparameters:
    # Activation (categorical)
    if output[0] < 0.25:
        activation = "custom"
    elif output[0] < 0.5:
        activation = "relu"
    elif output[0] < 0.75:
        activation = "tanh"
    else:
        activation = "sigmoid"

    # Learning rates on log scale:
    init_lr = 10 ** (np.log10(1e-1) + output[1]*(np.log10(1)-np.log10(1e-1)))
    final_lr = 10 ** (np.log10(1e-5) + output[2]*(np.log10(1e-2)-np.log10(1e-5)))

    # Epoch counts:
    warmup_epochs = int(50 + output[3]*(100-50))
    decay_epochs  = int(50 + output[4]*(200-50))

    # Number of layers:
    n_layers = int(3 + output[5]*(6-3))

    # Hidden layer sizes:
    n_hidden = []
    for i in range(6):
        units = int(128 + output[6+i]*(1024-128))
        # Quantize to the nearest multiple of 128:
        units = int(round(units/128)*128)
        n_hidden.append(units)
    # Use only the first n_layers
    n_hidden = n_hidden[:n_layers]

    # Batch size:
    bs = int(1000 + output[12]*(10000-1000))
    bs = int(round(bs/1000)*1000)

    # Define a simple learning rate schedule:
    decay_rate = (final_lr / init_lr) ** (1 / decay_epochs)
    def lr_schedule(step):
        return jnp.where(
            step < warmup_epochs,
            init_lr,
            init_lr * (decay_rate ** (step - warmup_epochs))
        )

    # Setup the model and optimizer:
    model = Emulator(hidden_layers=n_hidden, output_dim=n_modes, activation=activation)
    rng = jax.random.PRNGKey(0)
    dummy_input_jax = jnp.ones((1, dim_X))
    params = model.init(rng, (dummy_input_jax - X_mean_jax) / X_std_jax)
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(params)

    # Create model configuration tuple: (n_hidden, output_dim, activation)
    model_config = (tuple(n_hidden), n_modes, activation)

    best_val_loss = np.inf
    patience = 100
    early_stopping_counter = 0
    total_epochs = warmup_epochs + decay_epochs

    # -----------------------------------------------------------------------------
    # Initialize WandB logging with genome and generation info.
    # -----------------------------------------------------------------------------
    run = wandb.init(
        project="Emulator-NEAT-Evolution-Trials",
        config={
            "activation": activation,
            "init_lr": init_lr,
            "final_lr": final_lr,
            "warmup_epochs": warmup_epochs,
            "decay_epochs": decay_epochs,
            "n_hidden": n_hidden,
            "batch_size": bs,
            "generation": current_generation  # Log the generation number
        },
        name=f"Gen {current_generation} Genome {genome.key}",
        reinit=True
    )

    for epoch in range(total_epochs):
        epoch_losses = []
        for batch_X, batch_Y in get_batches(X_train, Y_train, bs):
            batch_X_jax = jnp.array(batch_X)
            batch_Y_jax = jnp.array(batch_Y)
            params, opt_state, loss_val = train_step(optimizer, params, opt_state, batch_X_jax, batch_Y_jax, model_config)
            epoch_losses.append(loss_val.item())
        avg_train_loss = np.mean(epoch_losses)
        val_loss = compute_loss(params, jnp.array(X_val), jnp.array(Y_val), model_config)
        val_loss_item = val_loss.item()

        # Calculate current learning rate for the epoch.
        current_lr = float(lr_schedule(epoch))

        # Log metrics including the current epoch.
        wandb.log({
            "Epoch": epoch,
            "Log Training Loss": jnp.log10(avg_train_loss),
            "Log Validation Loss": jnp.log10(val_loss_item),
            "Learning Rate": current_lr
        })

        # Early stopping check:
        if val_loss_item < best_val_loss:
            best_val_loss = val_loss_item
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                break

    # Finish the WandB run.
    wandb.finish()

    # Define fitness as the negative best validation loss (since NEAT maximizes fitness)
    fitness = -best_val_loss
    return fitness

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

# =============================================================================
# NEAT Evolutionary Loop Runner
# =============================================================================

def run_neat(config_file, n_generations=50):
    """
    Load the NEAT configuration, create the population, and run the evolution.
    """
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    population = neat.Population(config)
    
    # Add reporters to show progress in the terminal.
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    # Add the custom WandB reporter to capture generation number.
    population.add_reporter(WandbReporter())

    # Run NEAT for the specified number of generations.
    winner = population.run(eval_genomes, n_generations)

    # Save the winning genome.
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Best genome:\n", winner)
    return winner

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Provide the NEAT configuration file (e.g., "neat_config.ini")
    winner = run_neat("/content/drive/MyDrive/neat_config.ini", n_generations=50)