# Exploring Properties of Bioplausible Working Memory

## Authors
- **Willian Soares Girão** -  PhD student, University of Groningen, Faculty of Science and Engineering, Bio-inspired 
  Circuits & Systems
- **Thomas Tiotto** - PhD student, University of Groningen, Faculty of Science and Engineering, Cognitive Modelling

## References
- Mongillo, G., Barak, O. & Tsodyks, M. Synaptic Theory of Working Memory. Science 319, 1543–1546 (2008)
- Jug & Florian. On Competition and Learning in Cortical Structures (2012)
- Lundqvist, M., Rehn, M., Djurfeldt, M. & Lansner, A. Attractor dynamics in a modular network model of neocortex. Netw Comput Neural Syst 17, 253–276 (2009)
- Pals, et al. A functional spiking-neuron model of activity-silent working memory in humans based on calcium-mediated short-term synaptic plasticity. Plos Comput Biol 16, e1007936 (2020)
- Adaptations of original work from Ankatherin Sonntag's Master's thesis on the Chicca synaptic learning rule

## Usage
- `network_dynamics > RCN > rcn.py`: Runs a recurrent competitive network (RCN) learning to sustain 
  attractor activity for two stimuli.  Output plots and data are saved to the network_results folder.
- `graph_analysis > network_visualisation.py`: Runs a recurrent competitive network (RCN) learning to sustain 
  attractor activity for two stimuli.  Builds and displays a graph from the RCN in order to characterise the network 
  topology.  Output files can be saved by calling the `save_graph_results()` function
- `graph_analysis > network_visualisation.py`: Loads and displays a pre-existing RCN graph in order to characterise the 
  network topology.
