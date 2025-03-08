import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

# Simulation parameters
defaultclock.dt = 0.1*ms  # Time step
duration = 500*ms  # Total simulation time
bin_size = 15*ms  # Binning interval

# Input current (step current)
I_ext = 20 * nA  # Decreased from 200 nA back to 50 nA

# ------------------- LIF Model -------------------
eqs_LIF = '''
dv/dt = (-(v - EL) + I*Rm)/tau : volt  # Rewritten to make units consistent
I : amp
Rm = 1*Mohm : ohm  # Membrane resistance
tau = 20*ms : second
EL = -65*mV : volt
'''

neuron_LIF = NeuronGroup(1, eqs_LIF, method='euler',
                         threshold='v > -50*mV', reset='v = EL', refractory=2*ms)  # Added refractory period
neuron_LIF.v = -65*mV
neuron_LIF.I = I_ext

spikemon_LIF = SpikeMonitor(neuron_LIF)

# Add state monitors to record voltage traces
statemon_LIF = StateMonitor(neuron_LIF, 'v', record=True)

# ------------------- Izhikevich Model -------------------
eqs_IZH = '''
dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms : 1
du/dt = (a*(b*v - u)) / ms : 1
I : 1
a : 1
b : 1
c : 1
d : 1
'''

neuron_IZH = NeuronGroup(1, eqs_IZH, method='euler',
                         threshold='v > 30', reset='v = c; u += d')
neuron_IZH.v = -65
neuron_IZH.u = -14
neuron_IZH.a = 0.02
neuron_IZH.b = 0.2
neuron_IZH.c = -65
neuron_IZH.d = 8
neuron_IZH.I = 15

spikemon_IZH = SpikeMonitor(neuron_IZH)

# Add state monitors to record voltage traces
statemon_IZH = StateMonitor(neuron_IZH, 'v', record=True)

# ------------------- Run Simulation -------------------
run(duration)

# ------------------- Binning Spikes -------------------
def bin_spikes(spikemon, duration, bin_size):
    bins = np.arange(0, duration/ms, bin_size/ms)
    spike_counts, _ = np.histogram(spikemon.t/ms, bins)
    return bins[:-1], spike_counts

bins_LIF, spikes_LIF = bin_spikes(spikemon_LIF, duration, bin_size)
bins_IZH, spikes_IZH = bin_spikes(spikemon_IZH, duration, bin_size)

# ------------------- Plot Results -------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Raster Plot
axes[0].scatter(spikemon_LIF.t/ms, np.ones(len(spikemon_LIF.t)), label="LIF", color='g', s=10)
axes[0].scatter(spikemon_IZH.t/ms, np.ones(len(spikemon_IZH.t))*2, label="Izhikevich", color='r', s=10)
axes[0].set_title("Spike Raster Plot")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Neuron Model")
axes[0].legend()

# Binned Spikes
axes[1].bar(bins_LIF, spikes_LIF, width=bin_size/ms, alpha=0.5, label="LIF", color='g')
axes[1].bar(bins_IZH, spikes_IZH, width=bin_size/ms, alpha=0.5, label="Izhikevich", color='r')
axes[1].set_title("Binned Spike Counts (15 ms bins)")
axes[1].set_xlabel("Time (ms)")
axes[1].set_ylabel("Spike Count")
axes[1].legend()

# Voltage Traces
axes[2].plot(statemon_LIF.t/ms, statemon_LIF.v[0]/mV, label="LIF", color='g')
axes[2].plot(statemon_IZH.t/ms, statemon_IZH.v[0], label="Izhikevich", color='r')
axes[2].set_title("Voltage Traces")
axes[2].set_xlabel("Time (ms)")
axes[2].set_ylabel("Membrane Potential (mV)")
axes[2].legend()

plt.tight_layout()
plt.show()
