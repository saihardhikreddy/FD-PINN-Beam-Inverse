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
The beam is analysed in the frequency domain: the network takes the position $x \in [0, 10]$ and the excitation frequency $f \in [0, 0.5]$ and returns the complex deflection $\hat{w}(x, f) = w_R + i\,w_I$. With $\omega = 2\pi f$, `FD-PINN_Optimized.py` enforces

$$EI \frac{d^4 \hat{w}}{dx^4} - m\,\omega^2 \hat{w} + 2 i\,k_s\,\omega^2 \hat{w} = \frac{p_0}{v}\, e^{-i \omega x / v}$$

as two real equations (its real and imaginary parts, `eq_ur` and `eq_ui`), where:
* $EI = 1000 + CC$: flexural rigidity; $CC$ is the trainable unknown the inverse problem solves for
* $m = 1$: mass per unit length
* $k_s = 0.025$: coefficient of the $2 i k_s \omega^2 \hat{w}$ term (called the foundation coefficient in the script). Because it is imaginary, it acts like damping and couples $w_R$ and $w_I$
* $p_0 = 20$, $v = 1$: magnitude and speed of the moving point load, whose Fourier transform gives the right-hand side
* $L = 10$: beam length

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
The L-BFGS optimizer reduced the combined weighted loss to `1.58e+02` over 3000 epochs (see `loss.dat`). The bundled predictions reproduce the 41 sensor readings with an RMSE of 0.36 for $w_R$ and 0.79 for $w_I$, against readings up to 5.2 (`python plot_results.py` prints these). The discovered $CC$ is written to `variables.dat` during training; that file is not included here, so rerun `FD-PINN_Optimized.py` to see the identified stiffness.

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
Reshapes the saved predictions into a frequency × position grid and draws the displacement field. It then prints how closely the predictions match the 41 sensor readings in `data_inverse.txt` (RMSE 0.36 for $u_r$ and 0.79 for $u_i$, against readings up to 5.2), and how each loss term (PDE residuals, sensor fits, boundary conditions) fell during training, from `loss.dat`. It needs only numpy and matplotlib.

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
