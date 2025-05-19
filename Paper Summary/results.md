## 1. Overview

This work presents a detailed comparison of emulators for the linear CAMB matter power spectrum $P(k,z)$, contrasting a deep dense network (“baseline emulator”) against a compact architecture discovered via neuroevolution (NEAT). Two sampling strategies—emulating each redshift independently (“z method”) versus a 2D logarithmic–linear grid over $(k,z)$ (“grid method”)—are evaluated in terms of speed and accuracy. The NEAT-optimized model matches baseline precision while drastically reducing computational cost, and both emulators are benchmarked in a Limber integral application for convergence power spectra. All timings in this document are performed on the University of Cambridge High Performance Cluster, icelake partition, and all averages and standard deviations are calculated over 10,000 iterations.

## 2. Baseline Emulator: Sampling Methods & Architecture

### 2.1 Data and Pre-/Post-Processing

* **Spectra**: linear CAMB $P(k,z)$ sampled at $k \in [10^{-4}, 10^{1}]\,h\,\mathrm{Mpc}^{-1}$ (200 log-spaced points) and $z \in [10^{-5}, 5]$ 
 (10 logarithmically spaced points for grid method).
* **Preprocessing**: take $\log_{10}P$, then standardise (subtract mean, divide by stdev) over full training set and standardising all intput parameters (subtrace mean, divide by stdev)
* **Postprocessing**: un-standardise outputs using stored mean and variance.

### 2.2 Network Architecture

* **Layers**: 4 hidden layers, each 1,024 units.
* **Activation**:

  $$
    f(x)=x\bigl[\beta + \sigma(\alpha x)(1-\beta)\bigr],
  $$

  where $\alpha,\beta$ are trainable scalars.
* Each node has a bias and weight which can be trained, the initial weight is picked from $\sim\mathcal{N}(0,10^{-3})$ and all biases are initially 0
* 2 vectors are initialised, which are the length of the number of features (nodes) in the neural network, with the value in each position being picked randomly from $\sim\mathcal{N}(0,1)$. So each node has its own, unique activation function.
* **Outputs**:

  * **Z method**: 200 output nodes per pass (power spectrum at one $z$ at a time).
  * **Grid method**: 2,000 output nodes per pass (power spectrum at 10 $z$ simultaneously, with smallest $z$ ($10^{-5}$) power spectrum first and largest $z$ ($5$) power spectrum last).

* **Inputs**:
  * **Z method**: A set of parameters of the universe (whose values are randomly picked) which are needed to calculate the power spectrum *and* a random time-scale, $z$, 7 inputs in total 
  * **Grid method**: A set of parameters of the universe (whose values are randomly picked) which are needed to calculate the power spectrum *without* a time-scale, 6 inputs in total.
  * **Universe Parameters needed for power spectrum**: 
    * 

### 2.3 Training Setup

* **Loss**: mean-squared error over all $(k,z)$ points.
* **Optimiser**: Optax Adam

  * init\_lr = 0.05
  * final\_lr = 1e-4
* **LR schedule**: 150 warmup epochs + 450 decay epochs, where

$$
\text{decay\_rate}
= \biggl(\frac{\text{final\_lr}}{\text{init\_lr}}\biggr)^{1/\text{decay\_epochs}}
$$


  and step size = batch\_size / n\_training.
* **Batch size**: 10,000
* **Interpolation**: JAX `map_coordinates` in $(\log k,\log z)$ for arbitrary $(k,z)$.

### 2.4 Performance: Z vs. Grid

| Method      | Total time to Emulate Grid and Interpolate for a single value (s)             | Mean % Diff vs. CAMB at random k, z and Universe parameters  |
| ----------- | ----------------------- | ---------------------- |
| CAMB        | $2.96045216\pm0.094385$ |           N/A          |
| Z method    | $0.16891351\pm0.007754$ | $12.98351\pm9.243690$% |
| Grid method | $0.02179045\pm0.007500$ | $7.43212\pm3.817599$%  |

> **Conclusion:** The grid method is \~7× faster and \~2× more accurate than the z method and \~130x faster than using CAMB to calculate the grid; and will be adopted for subsequent experiments. This is because the grid method only has to emulate once whereas the z method has to emulate 10 times. Accuracy of grid method is better because it's only trained on the z's it needs to emulate, whereas the z method is also trained on z's which are never emulated, so are useless training points. The maximum percentile difference on the grid method is \~2.5%, whereas it's \~5% for the z method.

## 3. Neuroevolution-Optimised Emulator (NEAT)

### 3.1 Evolutionary Search Setup

NeuroEvolution of Augmenting Topologies (NEAT) is an evolutionary algorithm designed to optimise neural network architectures and weights simultaneously. It starts with a simple initial population of neural networks and progressively introduces complexity through evolutionary processes such as mutation (e.g., adding neurons or connections) and crossover. Each candidate network's performance (fitness) is evaluated—in this case, via validation loss—to guide selection and evolution across generations. NEAT effectively discovers compact and efficient architectures customised for the task at hand. The original NEAT implementation and detailed documentation are publicly available in the official repository at https://github.com/CodeReclaimers/neat-python. The best architecture was determined by having the smallest validation loss after being trained.

A NEAT search was conducted over the following gene space:

| Gene idx | Parameter       | Domain                        |
| :------: | :-------------- | :---------------------------- |
|     0    | Activation      | {custom, relu, tanh, sigmoid} |
|     1    | Initial LR      | $\log_{10}[10^{-5},\,1]$      |
|     2    | Final LR        | $\log_{10}[10^{-5},\,1]$      |
|     3    | Warmup epochs   | \[10, 300] (integer)          |
|     4    | Decay epochs    | \[10, 300] (integer)          |
|     5    | # Hidden layers | \[1, 4] (integer)             |
|   6–11   | Units per layer | \[16, 512] (step 16)          |
|    12    | Batch size      | \[100, 2000] (step 100)       |

* **Population**: 15 networks per generation
* **Generations**: 50 (early stop on single specimen if no validation loss improvement for 100 epochs)
* **Data per candidate**: 10% of training set
* **Training**: losses tracked in Weights & Biases.

### 3.2 Best-Found Architecture

* **Activation**: custom
* **Hidden layers**: \[256]
* **Batch size**: 100
* **Warmup epochs**: 10
* **Decay epochs**: 299
* **Init LR**: 0.100000
* **Final LR**: 0.760228

### 3.3 Performance & Retraining

* **Retrained on 10% data**:

| Architecture | Rough maximum error compared to CAMB data       | Rough minimum error compared to CAMB data                         |
| :-----------------------: | :-------------- | :---------------------------- |
|     Original 4x\[1024]    | 0.25%      | 0.05% |
|     NEAT \[256]   | 0.25%      | 0.05%      |


* **Final retraining on 80% data** (to smooth LR schedule spikes):

  * Warmup = 200 epochs, Decay = 300 epochs
  * Init LR = 0.01, Final LR = 0.001
  * **Total time to Emulate and Interpolate for a single value**: $0.007499\pm0.002937$ s
  * **Mean % Diff vs. CAMB at random k, z and Universe parameters**: $7.42109\pm3.81302$%

> **Conclusion:** The neural network using NEAT architecture has the same accuracy as the original architecture, but is ~2.5x faster as there are fewer nodes to calculate when emulating. Interestingly, despite the percentage difference between validation data and emulated data being an order of magnitude smaller than the original architecture, the percentage difference on interpretation value is the same within errors.

## 4. Convergence Power Spectrum via Limber Integral

### 4.1 Implementation Details

Compute convergence spectrum

$$
  C_\ell^{\kappa\kappa} = \int \frac{W^2(\chi)}{\chi^2} \; P\Bigl(k=\tfrac{\ell+1/2}{\chi},\,z(\chi)\Bigr) \, d\chi
$$

* **Window** $W(\chi) = \frac{3 H_0^2 \,\Omega_m}{2 c^2}\,\frac{\chi}{a(\chi)}\,\frac{\chi_* - \chi}{\chi_*}
, a = \frac{1}{1+z},\ \chi = \frac{D_A}{1+z}$ where $\chi^{*}$ and $D_{A}$ are calculated using CLASS
* **Integration**: Simpson’s rule in JAX
* **High-$z$ fallback** ($z>5$): $P(z)=P(5)\times\bigl((1+5)/(1+z)\bigr)^2$
* **Three methods of calculating $C_\ell^{\kappa\kappa}$**:
    * Performing integral and using neural network to calculate the value of $P(k, z)$ at every point, using 200 points
    * Performing integral and using CAMB to calculate the value of $P(k, z)$ at every point, using 200 points
    * Using CLASS built in lensing

For the first 2 methods the lensing power spectrum is calculated by performing the integral above at every $\ell$ required. The integral is performed by calculating the value of the integrand at every z and using Simpson's rule to find the integral. If it's performed using CAMB P(k) then it explicitly calculates $P(k,z)$ for all $\ell$ and z, whereas if it is done using the z method then the value of $P(k,z)$ is emulated at all $\ell$ and z. If the integral is performed using the grid method then $P(k,z)$ is emulated once and interpolated to find values for all $\ell$ and z, then the values are just "picked" from this grid when needed.

### 4.2 Runtime & Accuracy for Integrating and Using Neural Networks to calculate linear $P(k, z)$

n_x is the number of chunks the integral is broken up into, and is 200 for all Limber integral calculations (i.e. for the first 2 of 3 methods).

| Emulator           | Lensing Spectrum Compute Time (s)    | $r^2$ vs. Integrating and using CAMB for $P(k, z)$ |
| ------------------ | --------------------- | ----------------- |
| Baseline 4×\[1024] | $0.040434\pm0.023177$ | 0.99996415        |
| NEAT \[256]        | $0.030220\pm0.018555$ | 0.99996517        |

> **Conclusion:** The neural network using NEAT architecture has the same accuracy as the original architecture, but is faster as there are fewer nodes to calculate when emulating.

## 5. Enhanced Sampling & Non-linear Spectra

### 5.1 Log–Lin Grid (Linear $P(k, z)$)

* **Sampling grid**: 50 $z$ values (40 log-spaced $10^{-5}$–5, 10 linearly-spaced 7.5–30)
* **Architecture**: NEAT \[256]
* **Total time to Emulate and Interpolate for a single value**: $0.010430\pm0.0072539$ s
* **Mean % Diff vs. CAMB at random k, z and Universe parameters**: $0.7335497\pm0.564757$%

### 5.2 Log–Lin Grid (Non-linear $P(k, z)$)

* **Sampling grid**: 50 $z$ values (40 log-spaced $10^{-5}$–5, 10 linearly-spaced 7.5–30)
* **Architecture**: NEAT \[256]
* **Total time to Emulate and Interpolate for a single value**: $0.01249396\pm0.0094293$ s
* **Mean % Diff vs. CAMB at random k, z and Universe parameters**: $0.8698962\pm0.7990587$%

### 5.3 Limber Integral Plotting Times for Integral in 200 Chunks

| Method           | Lensing Spectrum Compute Time (s)    | $r^2$ vs. Integrating and using CAMB for $P(k, z)$ |
| ------------------ | --------------------- | ----------------- |
| JAX integration and linear power emulator | $0.033180\pm0.0019962$ | 0.99999625        |
| JAX integration and CAMB linear power | $4.038866\pm0.0362528$ | N/A      |
| Jax integration and non-linear power emulator | $0.029563\pm0.0021856$ | 0.99999704      |
| JAX integration and CAMB non-linear power | $4.664761\pm0.0318860$ | N/A      |
| CLASS built in lensing (non-linear) | $0.078605\pm0.0064085$ |  N/A   |


> **Conclusion:** Using a neural network to emulate the power spectrum in the integral for the lensing spectrum is \~10x quicker than using the CLASS P(k) for integration, for both non-linear and linear power spectra. The 99th percentile percentage difference for non-linear and linear lensing spectra is maximum at $\ell\approx2000$ at \~1%. This can be reduced by using more training data. Changing the number of chunks the integral is broken up into doesn't change the accuracy much.

## 6. Conclusions & Future Work

* The **grid sampling** strategy outperforms the z method by \~7× in speed and \~2× in interpolation accuracy and is \~135x quicker than CAMB.
* A **NEAT-derived \[256] model** matches a 4×1,024 network’s precision with \~2.5× faster inference and 3× fewer parameters.
* **Log–Lin grids** for linear and non-linear training achieve sub-percent errors up to $\ell=2000$.
* **Emulating power spectra using the neural network** reduces time to calculate the lensing spectrum by \~150x (compared to CAMB integration method), making it more applicable for reinforcement learning techniques.

**Future directions**:

* Integrate the NEAT loglin emulator into MCMC parameter-inference pipelines for full cosmological analyses.
* Extend to galaxy power spectra, redshift-space distortions, and higher-order statistics.
* Release code, training logs, and data for community reproducibility.

**Figures to embed:**

* `z_method_percentile_plot.png`
* `grid_percentile_plot_1M_OG_Architecture.png`
* `grid_percentile_plot_1M_NEAT_Architecture.png`
* `limber_C_ell_comparison_linear.png`
* `limber_C_ell_comparison_nonlinear.png`
* `LogLin_error_vs_ell_linear_200_chunks.png`
* `LogLin_error_vs_ell_nonlinear_200_chunks.png`
* `LogLin_error_vs_ell_linear_256_chunks.png`
* `LogLin_error_vs_ell_nonlinear_256_chunks.png`

All scripts, data, and logs are available in the project repository for full reproducibility.
