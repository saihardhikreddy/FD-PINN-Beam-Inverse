import deepxde as dde
import numpy as np
import math
import tensorflow as tf
import os

# ==========================================
# 1. SETUP PRECISION (Strict float64 for structural math)
# ==========================================
dde.config.real.set_float64()
dde.config.set_default_float("float64")

# ==========================================
# 2. LOAD VERIFIED SENSOR DATA
# ==========================================
data = np.loadtxt('data_inverse.txt', dtype='float64')

# ==========================================
# 3. CONSTANTS & PARAMETERS
# ==========================================
num = data.shape[0]
PI = math.pi
m = 1.0     # Mass per unit length
v = 1.0     # Velocity of moving load
p0 = 20.0   # Load magnitude
fmax = 0.5  # Time limit
L = 10.0    # Beam Length
ks = 0.025  # Foundation coefficient

# ==========================================
# 4. INVERSE PARAMETER (The hidden stiffness)
# ==========================================
CC = dde.Variable(1.0, dtype='float64')

# ==========================================
# 5. SPATIOTEMPORAL DOMAIN
# ==========================================
geom = dde.geometry.Interval(0, L)
timedomain = dde.geometry.TimeDomain(0, fmax)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# ==========================================
# 6. PHYSICS (Euler-Bernoulli PDE)
# ==========================================
def pde(x, y):
    # 4th-order spatial derivatives
    ur_xxxx = dde.grad.hessian(dde.grad.hessian(y, x, component=0, i=0, j=0), x, i=0, j=0)
    ui_xxxx = dde.grad.hessian(dde.grad.hessian(y, x, component=1, i=0, j=0), x, i=0, j=0)
    
    # Dynamic Stiffness 
    EI = 1000.0 + CC 

    # Governing Equations for forced vibration
    eq_ur = EI*ur_xxxx - 2*ks*y[:,1:2]*(x[:, 1:2]*2*PI)**2 - m*y[:,0:1]*(x[:, 1:2]*2*PI)**2 - p0/v*(tf.cos(x[:, 0:1]*x[:, 1:2]*2*PI/v))
    eq_ui = EI*ui_xxxx + 2*ks*y[:,0:1]*(x[:, 1:2]*2*PI)**2 - m*y[:,1:2]*(x[:, 1:2]*2*PI)**2 + p0/v*(tf.sin(x[:, 0:1]*x[:, 1:2]*2*PI/v))
    
    return [eq_ur, eq_ui]

# ==========================================
# 7. BOUNDARIES & SENSOR DATA INTEGRATION
# ==========================================
def boundary_l(x, on_boundary): return on_boundary
def fun_ur_xx(x, y, _): return dde.grad.hessian(y, x, component=0, i=0, j=0)
def fun_ui_xx(x, y, _): return dde.grad.hessian(y, x, component=1, i=0, j=0)

# Simple Supports
bc1 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_l, component=0)
bc2 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_l, component=1)
bc3 = dde.icbc.OperatorBC(geomtime, fun_ur_xx, boundary_l)
bc4 = dde.icbc.OperatorBC(geomtime, fun_ui_xx, boundary_l)

# Sensor point mapping
observe_x = np.vstack((data[:,0], data[:,1])).T
observe_y1 = dde.icbc.PointSetBC(observe_x, data[:,2].reshape(num,1), component=0)
observe_y2 = dde.icbc.PointSetBC(observe_x, data[:,3].reshape(num,1), component=1)

# Training Anchors
xx, yy = np.meshgrid(np.linspace(0, L, 21), np.linspace(0, fmax, 41))
AN_X = np.concatenate((np.concatenate((xx.flatten()[:, None], yy.flatten()[:, None]), 1), observe_x), 0)

pde_data = dde.data.TimePDE(geomtime, pde, [observe_y1, observe_y2, bc1, bc2, bc3, bc4], anchors=AN_X)

# ==========================================
# 8. OPTIMIZED ARCHITECTURE
# MODIFICATION: 4 layers, 50 neurons, 'sin' activation for oscillatory physics
# ==========================================
net = dde.maps.FNN([2] + [50] * 4 + [2], "sin", "Glorot uniform")

net.apply_feature_transform(lambda x: tf.concat([x[:,0:1]/L, x[:,1:2]/fmax], axis=1))
net.apply_output_transform(lambda x, y: y * 10)

model = dde.Model(pde_data, net)
dde.optimizers.set_LBFGS_options(ftol=0, gtol=1e-20, maxiter=3000)

# ==========================================
# 9. OPTIMIZED TRAINING WEIGHTS
# MODIFICATION: Increased boundary condition weights (1e3, 1e2)
# ==========================================
model.compile("L-BFGS-B", 
              loss_weights=[1, 1, 1e2, 1e1, 1e3, 1e2, 1e6, 1e7], 
              external_trainable_variables=CC)

variable = dde.callbacks.VariableValue(CC, period=100)

# ==========================================
# 10. EXECUTE TRAINING
# ==========================================
losshistory, train_state = model.train(
    epochs=3000, 
    display_every=100, 
    callbacks=[variable], 
    model_save_path="./model/trained-model-optimized"
)

# ==========================================
# 11. SAVE OPTIMIZED RESULTS
# ==========================================
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Generate specific outputs for the optimized model
x_star = np.linspace(0, L, 101)
f_star = np.linspace(0, fmax, 41)
X1_star, F1_star = np.meshgrid(x_star, f_star)
X = np.vstack((X1_star.flatten(), F1_star.flatten())).T

pred = model.predict(X)
np.savetxt('ur_pred_optimized.txt', pred[:, 0])
np.savetxt('ui_pred_optimized.txt', pred[:, 1])

print("Optimized Training Complete. Results saved successfully.")