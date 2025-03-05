import os
import sys
os.environ["OMP_NUM_THREADS"]= "76"
import camb
from camb import model, initialpower
import numpy as np
from pyDOE import lhs
import matplotlib.pyplot as plt
import threading
import time

# Start timing
start_time = time.perf_counter() 


# Define the possible parameter ranges
param_bounds = {
    "H0": (65, 75),
    "ombh2": (0.02, 0.025),
    "omch2": (0.1, 0.15),
    "logA": (2.5, 3.5),
    "ns": (0.94, 1.0),
    "z": (0, 5)
}


# Define the number of sets of random parameters
num_samples = 1000


# Generate Latin Hypercude samples in normalised unit cube
lhs_samples = lhs(len(param_bounds), samples=num_samples)

# Scale the sample to the parameter bounds
scaled_params = []
for i, (param, bounds) in enumerate(param_bounds.items()):
    lower, upper = bounds
    scaled_params.append(lhs_samples[:, i] * (upper - lower) + lower)


# Transpose to get a list of dictionaries for each example
sampled_params = [
    {param: scaled_params[j][i] for j, param in enumerate(param_bounds)}
    for i in range(num_samples)
]


# Function to compute the power spectrum of a given set of parameters
def compute_power_spectrum(H0, ombh2, omch2, logA, ns, z):

    # Convert logA to As
    As = np.exp(logA) / 1e10

    # Set up CAMB parameters
    pars=camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, As=As, ns=ns)
    pars.set_matter_power(redshifts=[z], kmax=10.0)

    # Calculate the power spectrum from the parameters given
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=10, npoints = 200)

    return pk[0]


# Array for storing the power spectra for each set of parameters, npoints x num samples
spectra = []
param_dict = {  # Dictionary to store each parameter as a separate key
    "H0": [],
    "ombh2": [],
    "omch2": [],
    "logA": [],
    "ns": [],
    "z": []
}


'''
Loops over the number of sets of parameters and calculates the power spectrum for each set, adding the 
pk and kh values to the dictionary and making a new row of the spectra array for each set of pk values
'''
for i, params in enumerate(sampled_params):
    H0, ombh2, omch2, logA, ns, z = params["H0"], params["ombh2"], params["omch2"], params["logA"], params["ns"], params["z"]
    pk = compute_power_spectrum(H0, ombh2, omch2, logA, ns, z)
    
    # Append values to their respective lists in the dictionary
    param_dict["H0"].append(H0)
    param_dict["ombh2"].append(ombh2)
    param_dict["omch2"].append(omch2)
    param_dict["logA"].append(logA)
    param_dict["ns"].append(ns)
    param_dict["z"].append(z)

    spectra.append(pk)

    # Print progress every 100 iterations
    if (i + 1) % 100 == 0:
        print(f"Generated {i + 1} pk values")


'''
Save the power spectra and parameters data in float32
'''
# Save in correct place with correct name
output_dir = "/home/jam249/rds/dlproject/logs"
task_id = sys.argv[1] if len(sys.argv) > 1 else "0"

# Convert to float32
spectra_fp32 = np.array(spectra, dtype=np.float32)  
for key in param_dict:
    param_dict[key] = np.array(param_dict[key], dtype=np.float32) 

# Save the data
np.save(f'{output_dir}/z_method_pk_data_{task_id}.npy', spectra_fp32)
np.save(f'{output_dir}/z_method_params_data_{task_id}.npy', param_dict)

# End timing
end_time = time.perf_counter() 

print(f"Total execution time: {end_time - start_time:.4f} seconds")