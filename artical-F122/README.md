# Provably Probabilistic Safe Controller Synthesis for Vision-Based Neural Network Control Systems
This is an example project for the CARLA Emergency Braking Benchmark used in the paper.


## Contributions
* We propose a **provably probabilistic safe controller synthesis
framework(SafePVC)** for **vision-based neural network control systems**,
providing formal probabilistic safety guarantees.
* We develop a **learnable stochastic barrier certificate** which
leverages martingale theory and data-driven disturbance to
enable probabilistic verification with formally certified lower
bounds on safety
* We synthesize provably safe controllers under the **guidance
of stochastic barrier certificates** and iteratively refine both
the certificate and controller networks using counterexamples, improving efficiency and practicality
## Installation
### Requirements
* Python 3.8+
* pytorch 2.3.1
* CUDA-capable GPU
### Setup
We suggest setting up a conda virtual environment using the provided `environment.yml` file.
```bash
conda env create -f environment.yml
```
To enable the use of interval-based analysis tools, you will also need to set up the auto_LiRPA environment.
```bash
cd auto_LiRPA
pip install .
```

## Repository Structure
```
.
├── Aebs/ # CARLA Emergency Braking Benchmark
│   ├── cGAN/ # Observation Model Approximation
│   ├── connect/ # Connect to simulator to collect training data
│   ├── controller/ # Controller Training
│   ├── data/ # Downsampled Dataset
│   ├── system/ # NNCS Construction
│   └── VT/ # SafePVC Main Framework
├── auto_LIRPA/
├── cGAN/ # cGAN Training Framework
├── Combined_network/ # Combined NNCS Network
├── environment.yml
└── README.md
```

## Usage
To begin, an approximate observation model is required. The main training framework is located in `./Aebs/cGAN`, and a trained observation model is already available in `./Aebs/cGAN/mlp_supervised_ld4`.

Next, the controller needs to be trained. The controller performs state estimation and action prediction based on the images produced by the observation model. The `./Aebs/controller` directory contains the training code for both the state estimation module and the reinforcement learning framework used for action prediction. The trained models are saved in `./Aebs/controller/state_net_trained.pth` and `./Aebs/controller/best_model`.

Finally, after constructing the complete vision-based NNCS, a barrier certificate and a safety-assured controller can be generated using the following command:

```bash
python -m Aebs.VT.loop
```


