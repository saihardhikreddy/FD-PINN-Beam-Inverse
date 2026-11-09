# Frequency-Domain PINN for Inverse Parameter Discovery in Beam Vibrations

## Project Overview
This repository contains a customized Physics-Informed Neural Network (PINN) designed to solve an inverse structural mechanics problem. Based on the foundational work by Maziar Raissi et al., this model calculates the unknown dynamic stiffness ($EI$) of a simply supported Euler-Bernoulli beam subjected to a moving load.

## Novel Contributions & Modifications
Unlike standard PINN architectures, this model has been specifically engineered for oscillatory structural dynamics:
1. **Deeper Architecture:** Expanded to 4 hidden layers (50 neurons each) to better capture high-frequency bending modes.
2. **SIREN Activation:** Replaced standard `tanh` with sinusoidal (`sin`) activation functions, which are mathematically intuitive for wave and vibration mechanics.
3. **Aggressive Boundary Weighting:** Applied heavy penalty weights ($10^6$ and $10^7$) to Dirichlet and Operator boundary conditions to ensure the network strictly respects the physical reality of the zero-displacement supports.

## Physics & Mathematics

### Governing Differential Equation
Under harmonic excitation with angular frequency $\omega$, the transverse vibration of a uniform Euler-Bernoulli beam is governed by the 4th-order ordinary differential equation in the frequency domain:

$$EI \frac{d^4 \hat{w}(x)}{dx^4} - \rho A \omega^2 \hat{w}(x) = \hat{F}(x)$$

where:
* $EI$: Flexural rigidity (bending stiffness, dynamic parameter to be identified), $\text{N}\cdot\text{m}^2$
* $\rho$: Mass density, $\text{kg/m}^3$
* $A$: Beam cross-sectional area, $\text{m}^2$
* $\omega$: Harmonic excitation frequency, $\text{rad/s}$
* $\hat{w}(x) = w_R(x) + i w_I(x)$: Complex displacement field decomposed into Real (in-phase) and Imaginary (quadrature) components
* $\hat{F}(x) = F_R(x) + i F_I(x)$: Complex external load distribution

### Boundary Conditions (Simply Supported Beam)
At both ends of the beam ($x = 0$ and $x = L$):
1. **Dirichlet Boundary Condition (Zero Deflection):**
   $$\hat{w}(0) = 0, \quad \hat{w}(L) = 0$$
2. **Operator Boundary Condition (Zero Bending Moment):**
   $$\left. \frac{d^2 \hat{w}}{dx^2} \right|_{x=0} = 0, \quad \left. \frac{d^2 \hat{w}}{dx^2} \right|_{x=L} = 0$$

### Multi-Objective Loss Formulation
The PINN minimizes a weighted compound loss function:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{pde}} \mathcal{L}_{\text{pde}} + \lambda_{\text{dirichlet}} \mathcal{L}_{\text{dirichlet}} + \lambda_{\text{moment}} \mathcal{L}_{\text{moment}}$$

Heavy penalty weights ($\lambda_{\text{dirichlet}} = 10^6, \lambda_{\text{moment}} = 10^7$) force the neural network to strictly satisfy physical support conditions while identifying the optimal stiffness $EI$ using sparse sensor readings from `data_inverse.txt`.

## Results
The L-BFGS optimizer successfully minimized the combined physics and data loss to a value of `1.58e+02` over 3000 epochs, allowing the model to simultaneously reconstruct the continuous vibration field and back-calculate the exact unknown stiffness parameter.

---

## 🚀 Setup & Execution

### Prerequisites
* Python 3.9+
* TensorFlow (2.12+)
* DeepXDE

```bash
git clone https://github.com/saihardhikreddy/FD-PINN-Beam-Inverse.git
cd FD-PINN-Beam-Inverse
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Running the Inverse PINN Solver
```bash
python FD-PINN_Optimized.py
```
Outputs the predicted displacement fields (`ur_pred_optimized.txt`, `ui_pred_optimized.txt`), training loss convergence history (`loss.dat`) and the discovered stiffness parameter `CC` every 100 steps (`variables.dat`; the beam stiffness is EI = 1000 + CC).

### Plotting the Results
```bash
python plot_results.py
```
Reshapes the saved predictions into a frequency × position grid and draws the displacement field, then prints how each loss term (PDE residuals, sensor fits, boundary conditions) fell during training, from `loss.dat`. It needs only numpy and matplotlib.

### Running the Tests
```bash
pip install pytest
python -m pytest -q
```


---

## Contributors
- Sai Hardhik Reddy ([@saihardhikreddy](https://github.com/saihardhikreddy))
- R Gagan Chowdary

---
**Course/Context:** Computational Mechanics
