from scipy.integrate import odeint

import numpy as np

import matplotlib.pyplot as plt

#constants
cm = 1.0  #µF/cm3
gk = 36  #mS/cm2
gna = 120  #mS/cm2
gl = 0.3  #mS/cm2
vk = -12  #mV
vna = 115  #mV
vl = 10  #mV


#gated Variable

def alpha_m(V):
    return (25 - V) / (10 * (np.exp((25 - V) / 10) - 1))


def beta_m(V):
    return 4 * np.exp(-V / 18)


def alpha_h(V):
    return 0.07 * np.exp(-V / 20)


def beta_h(V):
    return 1 / (np.exp((30 - V) / 10) + 1)


def alpha_n(V):
    return (10 - V) / (100 * (np.exp((10 - V) / 10) - 1))


def beta_n(V):
    return 0.125 * np.exp(-V / 80)


#HH neuron model
def HH_Model(y, t):
    V, n, m, h = y
    Ik = gk * n ** 4 * (vk - V)
    Ina = gna * m ** 3 * h * (vna - V)
    Il = gl * (vl - V)
    dVdt = (1 / cm) * ((Ik + Ina + Il + Iext))
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    return [dVdt, dndt, dmdt, dhdt]


#Intial Conditions Problem 1
Iext = 5.2  #µA/cm2
VO = 0
nO = 0.35
mO = 0.06
hO = 0.6
t = np.linspace(0, 200, 10000)
y0 = [VO, nO, mO, hO]

#Solution
solution = odeint(HH_Model, y0, t)


# Plot
def Plots(t, solution):
    V, n, m, h = solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3]
    # Plot V
    plt.figure(figsize=(10, 6))
    plt.plot(t, V, label='Membrane Potential (V)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (V)')
    plt.title('Membrane Potential over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Plot n, m, and h
    plt.figure(figsize=(10, 6))
    plt.plot(t, n, label='n')
    plt.plot(t, m, label='m')
    plt.plot(t, h, label='h')
    plt.xlabel('Time (ms)')
    plt.ylabel('Value')
    plt.title('Gating Variables over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


Plots(t, solution)

#Intial Conditions Problem 2
Iext = 5.2
VO = 70
nO = 0.25
mO = 0.0
hO = 0.6
t = np.linspace(0, 200, 10000)
y0 = [VO, nO, mO, hO]
#sol
solution = odeint(HH_Model, y0, t)

Plots(t, solution)

#Intial Conditions Problem 3
Iext = 6.8
VO = 0
nO = 0.35
mO = 0.06
hO = 0.6
t = np.linspace(0, 200, 1000)
y0 = [VO, nO, mO, hO]
solution = odeint(HH_Model, y0, t)
Plots(t, solution)

#Intial Conditions Problem 4
Iext = 6.8
VO = 70
nO = 0.15
mO = 0.02
hO = 0.4
t = np.linspace(0, 200, 1000)
y0 = [VO, nO, mO, hO]
solution = odeint(HH_Model, y0, t)
Plots(t, solution)
