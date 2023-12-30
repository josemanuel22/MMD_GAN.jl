# MMD_GAN.jl

[![Build Status](https://github.com/josemanuel22/MMD_GAN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/josemanuel22/MMD_GAN.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Overview
`MMD_GAN.jl` is a Julia module implementing a [Maximum Mean Discrepancy (MMD) Generative Adversarial Network](https://arxiv.org/pdf/1705.08584.pdf). This module provides functionalities to train GAN models using MMD for measuring the discrepancy between the generated and real data distributions. It is designed for easy experimentation with different hyperparameters and model architectures. 

## Installation
To use MMD_GAN.jl, clone this repository into your local machine. Make sure you have Julia installed and set up on your system.

```bash
git clone git@github.com:josemanuel22/MMD_GAN.jl.git
```

## Usage
The module includes core functionalities to define hyperparameters, set up models, and train an MMD GAN. Key components include:

- `HyperParamsMMD`: A structure to define hyperparameters for the MMD GAN.
- `train_mmd_gan`: A function to train the MMD GAN using specified encoder, decoder, and generator models with given hyperparameters.

### Example
```julia
using MMD_GAN

# Define your models (encoder, decoder, generator)
# enc = ...
# dec = ...
# gen = ...

# Define hyperparameters
hparams = HyperParamsMMD(
    target_model = Normal(23.0f0, 1.0f0),
    noise_model = Normal(0.0f0, 1.0f0),
    # Other hyperparameters...
)

# Train the model
losses_gen, losses_dscr = train_mmd_gan(enc, dec, gen, hparams)
```

## Contributing
Contributions to MMD_GAN.jl are welcome. Please read our [contribution guidelines](CONTRIBUTING.md) for more details.

## License
This project is licensed under the [MIT License](LICENSE).

