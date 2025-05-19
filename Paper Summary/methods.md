### Methodology for the Replacement of Power Spectrum Calculations with Neural Networks

#### 1. Study Design Overview

The central objective of this project is to significantly accelerate cosmological power spectrum computations—traditionally conducted using computationally intensive numerical methods such as CAMB and CLASS—by developing efficient feed-forward neural network emulators. Rapid computation is essential for integration into a reinforcement learning algorithm aimed at parameter inference from observational data.

JAX was used to train the neural network because of 3 core ideas. These functions let you turn pure Python functions into fast, batches, multi-device workloads with a single deorator or wrapper:
* Composable function transorfmations
    * grad for automatic differentiation
    * jit for just-in-time compilation via XLA
    * vmap for automatic vecotrisation
    * pmap for parallelisation across multiple accelerators
* XLA compilation: traces your NumPy-style code and emits optimised, fused kernels for CPU/GPU/TPU - dramatically reducing kernel-launch overhead and memory traffic during both training and inference.
* Pure-functional style: all states (e.g.model parameters) are passed explicitly through functions making transformation like batching, differentiation and parallelisation more predictable and composable.

#### 2. Initial Two Types of Neural Networks

Two primary architectures were initially explored to emulate the cosmological power spectrum:

- **Z-Method:** 
    - Inputs: 7 parameters including cosmological parameters (e.g., curvature, dark matter proportion, primordial fluctuation amplitude) and the redshift (z).
    - Outputs: Power spectrum values computed at 200 logarithmically spaced wavenumbers (k) ranging from 10^-4 to 10.
    - Goal: Predict the power spectrum at arbitrary redshifts and cosmological parameters.

- **Grid Method:** 
    - Inputs: 6 cosmological parameters (excluding z).
    - Outputs: Power spectra at 10 fixed, logarithmically spaced redshifts (z) between 10^-5 and 5, with each spectrum consisting of 200 logarithmic wavenumbers (k). The total output is 2000 values per parameter set.
    - Goal: Predict the power spectrum only at specific predefined redshifts.

#### 3. Training Data Generation

Training and validation datasets were generated using CAMB on the University of Cambridge High-Performance Computing Cluster:
    - Parameters were sampled using Latin Hypercube Sampling (LHS) within physically meaningful cosmological ranges shown in "Parameter_Range_Fig".
    - Z-Method: 1 million samples, each with randomly chosen cosmological parameters and redshift.
    - Grid Method: 100,000 samples, each providing power spectra at 10 predefined redshifts.
    - Training utilised an 80%/20% split for training and validation respectively.
    - All spectra were generated using the CAMB respository.
    - Training data power spectra size: 1.34GB, Validation data power spectra size: 
343MB, Training data parameters size: 316kB, Validation data parameters size: 80kB
#### 4. Interpolation and Validation

Post-training, neural networks were evaluated using interpolation methods:
    - Linear interpolation in the redshift dimension was implemented for the Grid Method to enable predictions at arbitrary intermediate redshifts.
    - Speed and interpolation efficiency were benchmarked against direct numerical computations.
    - Interpolation grid shown in "Interpolation_Fig" where the red lines are values emulated by the neural network and the surfaces are interpolated values.

#### 5. Application to Angular Power Spectrum of Weak Lensing Convergence (PSWLC)

To demonstrate practical application, neural network predictions were integrated into the calculation of the PSWLC:
* The PSWLC, an integral of the power spectrum over redshift, was calculated using predictions from neural network models.
* Computational time efficiency was compared against traditional numerical methods (direct numerical integration using CAMB, CLASS, and numerical integration via power spectrum emulation).
* This approach enables rapid iteration within a reinforcement learning framework for cosmological parameter inference from observational data.

#### 6. Neural Network Hyperparameter Optimisation

The NEAT (NeuroEvolution of Augmenting Topologies) algorithm was employed for systematic hyperparameter optimisation of neural network architectures. Key hyperparameters tuned include:
    - Number of hidden layers and nodes per layer
    - Activation functions
    - Batch size
    - Learning rate schedule (initial learning rate, final learning rate, warm-up epochs, decay epochs)

Optimisation was guided by validation error, ensuring selection of the most robust and accurate model architecture.

#### 7. Extended Redshift Sampling and Accuracy Improvement

Following initial results, an extended grid method was implemented:
    - Increased redshift sampling from 10 to 50 points (40 logarithmically between 10^-5 and 5, plus 10 linearly between 5 and 30).
    - Neural networks trained with this richer sampling provided enhanced interpolation accuracy while maintaining computational efficiency.

#### 8. Transition to Nonlinear Power Spectrum

To address deviations observed at multipoles l roughly = 300 and beyond:
    - Neural networks were retrained using nonlinear power spectrum data generated via CAMB or CLASS.
    - Enhanced accuracy at higher multipoles significantly improved the model's relevance for realistic cosmological analyses, particularly in the nonlinear regime crucial for interpreting weak lensing convergence observations.

This comprehensive methodological refinement ensures accurate, rapid cosmological computations suited to advanced inference frameworks, offering a practical toolset for contemporary cosmological analysis.

**Figures to embed:**

* `log_z_visualisation.png`
* `loglin_z_visualisation.png`
* `interpolation_visualisation.png`
