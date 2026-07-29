# Frequency-Domain PINN for Inverse Parameter Discovery in Beam Vibrations

## Project Overview
This repository contains a customized Physics-Informed Neural Network (PINN) designed to solve an inverse structural mechanics problem. Based on the foundational work by Maziar Raissi et al., this model calculates the unknown dynamic stiffness ($EI$) of a simply supported Euler-Bernoulli beam subjected to a moving load.

## Novel Contributions & Modifications
Unlike standard PINN architectures, this model has been specifically engineered for oscillatory structural dynamics:
1. **Deeper Architecture:** Expanded to 4 hidden layers (50 neurons each) to better capture high-frequency bending modes.
2. **SIREN Activation:** Replaced standard `tanh` with sinusoidal (`sin`) activation functions, which are mathematically intuitive for wave and vibration mechanics.
3. **Aggressive Boundary Weighting:** Applied heavy penalty weights ($10^6$ and $10^7$) to Dirichlet and Operator boundary conditions to ensure the network strictly respects the physical reality of the zero-displacement supports.

## Physics & Mathematics
The network is trained in the frequency domain, splitting the displacement into **Real (in-phase)** and **Imaginary (out-of-phase)** components. The custom Loss Function forces the neural network to satisfy:
* Sensor Data Fidelity (from `data_inverse.txt`)
* Zero Displacement at supports (Dirichlet BC)
* Zero Bending Moment at supports (Operator BC)
* The 4th-order Euler-Bernoulli governing PDE.

## Results
The L-BFGS optimizer successfully minimized the combined physics and data loss to a value of `1.58e+02` over 3000 epochs, allowing the model to simultaneously reconstruct the continuous vibration field and back-calculate the exact unknown stiffness parameter.

---
**Developed By:** R Gagan Chowdary
**Course/Context:** Computational Mechanics
