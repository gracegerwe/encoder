from brian2 import *
import matplotlib.pyplot as plt

# Simulation parameters
duration = 105*ms  # Multiple of 15 ms for clean binning
dt = 0.1*ms  # Fine time step for neuron dynamics
defaultclock.dt = dt

# Time-binned encoder: Step input with 15 ms bins
# 105 ms = 7 bins of 15 ms: 2 off, 3 on, 2 off
stimulus = [0]*2 + [0.1]*3 + [0]*2  # 7 bins total
I_stim = TimedArray(stimulus*mV/ms, dt=15*ms)  # 15 ms bins

# --- Leaky Integrate-and-Fire (LIF) Model ---
LIF_eqs = '''
dv/dt = (-v/ms + I_stim(t)) : volt  # Leak time constant = 1 ms
'''
lif_neuron = NeuronGroup(1, LIF_eqs, threshold='v > 20*mV', reset='v = 0*mV', method='euler')
lif_neuron.v = 0*mV
lif_spike_mon = SpikeMonitor(lif_neuron)
lif_voltage_mon = StateMonitor(lif_neuron, 'v', record=True)

# --- Izhikevich Model ---
izh_eqs = '''
dv/dt = (0.04/mV * v**2 + 5*v + 140*mV - u + I_stim(t)*ms)/ms : volt
du/dt = (0.02 * (0.2*v - u))/ms : volt
'''
izh_neuron = NeuronGroup(1, izh_eqs, threshold='v > 30*mV', reset='v = -65*mV; u += 8*mV', method='euler')
izh_neuron.v = -70*mV
izh_neuron.u = -70*mV * 0.2
izh_spike_mon = SpikeMonitor(izh_neuron)
izh_voltage_mon = StateMonitor(izh_neuron, 'v', record=True)

# --- Hodgkin-Huxley (HH) Model ---
HH_eqs = '''
dv/dt = (-g_Na * m**3 * h * (v - E_Na) - g_K * n**4 * (v - E_K) - g_L * (v - E_L) + I_stim(t)*ms)/ms : volt
dm/dt = alpha_m * (1 - m) - beta_m * m : 1
dh/dt = alpha_h * (1 - h) - beta_h * h : 1
dn/dt = alpha_n * (1 - n) - beta_n * n : 1
alpha_m = 0.1/mV * 10*mV / (exp((v + 40*mV) / (10*mV)) - 1)/ms : Hz
beta_m = 4 * exp(v / (18*mV)) / ms : Hz
alpha_h = 0.07 * exp(v / (20*mV)) / ms : Hz
beta_h = 1 / (exp((v + 30*mV) / (10*mV)) + 1) / ms : Hz
alpha_n = 0.01/mV * 10*mV / (exp((v + 55*mV) / (10*mV)) - 1) / ms : Hz
beta_n = 0.125 * exp(v / (80*mV)) / ms : Hz
'''
g_Na, E_Na = 120*mS/cm**2, 50*mV
g_K, E_K = 36*mS/cm**2, -77*mV
g_L, E_L = 0.3*mS/cm**2, -54.4*mV
hh_neuron = NeuronGroup(1, HH_eqs, threshold='v > 0*mV', reset='v = -65*mV', refractory=2*ms, method='euler')
hh_neuron.v = -65*mV
hh_neuron.m, hh_neuron.h, hh_neuron.n = 0.05, 0.6, 0.32
hh_spike_mon = SpikeMonitor(hh_neuron)
hh_voltage_mon = StateMonitor(hh_neuron, 'v', record=True)

# Run simulation
run(duration)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# LIF
ax1.plot(lif_voltage_mon.t/ms, lif_voltage_mon.v[0]/mV, label='LIF')
ax1.plot(lif_spike_mon.t/ms, [20]*lif_spike_mon.num_spikes, 'ro', label='Spikes')
ax1.set_ylabel('Voltage (mV)')
ax1.legend()

# Izhikevich
ax2.plot(izh_voltage_mon.t/ms, izh_voltage_mon.v[0]/mV, label='Izhikevich')
ax2.plot(izh_spike_mon.t/ms, [30]*izh_spike_mon.num_spikes, 'ro', label='Spikes')
ax2.set_ylabel('Voltage (mV)')
ax2.legend()

# Hodgkin-Huxley
ax3.plot(hh_voltage_mon.t/ms, hh_voltage_mon.v[0]/mV, label='Hodgkin-Huxley')
ax3.plot(hh_spike_mon.t/ms, [0]*hh_spike_mon.num_spikes, 'ro', label='Spikes')
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Voltage (mV)')
ax3.legend()

plt.tight_layout()
plt.show()

# Print spike counts
print(f"LIF Spikes: {lif_spike_mon.num_spikes}")
print(f"Izhikevich Spikes: {izh_spike_mon.num_spikes}")
print(f"Hodgkin-Huxley Spikes: {hh_spike_mon.num_spikes}")