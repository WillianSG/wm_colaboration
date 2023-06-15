import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

# Set the random seed for reproducibility
np.random.seed(42)

# Number of neurons
num_neurons = 10

# Total simulation time (in milliseconds)
simulation_time = 1000

# Generate random spike times for each neuron
spike_times = [np.sort(np.random.uniform(0, simulation_time, np.random.randint(10, 20))) for _ in range(num_neurons)]

# Create the figure and axes
fig, ax = plt.subplots()

# Set the axis limits
ax.set_xlim(0, simulation_time)
ax.set_ylim(0, num_neurons)

# Set the labels and title
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Neuron')
ax.set_title('Raster Plot of Spike Times')

# Initialize an empty event plot
lines = ax.eventplot([], linelengths=0.8, colors='k')

# Update function for the animation
def update(frame):
    # Clear the previous spikes
    for line in lines:
        line.set_segments([])
    
    # Get the spike times up to the current frame
    frame_spike_times = [spike_time[spike_time <= frame] for spike_time in spike_times]
    
    # Update the plot with the new spike times
    lines = ax.eventplot(frame_spike_times, linelengths=0.8, colors='k')

# Function to initialize the animation
def init():
    return lines

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=simulation_time, init_func=init, blit=True)

# Save the animation as a GIF using imageio
ani.save('raster_plot.gif', writer='imageio', fps=30)

plt.show()
