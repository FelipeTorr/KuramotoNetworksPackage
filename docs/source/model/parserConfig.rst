Configuration file parser
===================================

The configuration file includes all the required parameters of the Kuramoto's model.
A *config* text file must contain the following parameters::

| #Config file for a Kuramoto model simulation
| experiment_name=Test
| #Network structure
| n_nodes=90
| struct_connectivity=../input_data/AAL_matrices.mat
| delay_matrix=../input_data/AAL_matrices.mat
| #Global parameters
| K=360.00
| mean_delay=0.018
| #Intrinsic/Natural frequencies
| nat_freqs=
| nat_freq_mean=40
| nat_freq_std=0.0
| #Time parameters 
| simulation_period=30
| dt=1e-4
| #Stim
| ForcingNodes=[0]
| StimTstart=22
| StimTend=25
| StimFreq=20 
| StimAmp=50
| #Stochastic parameters
| seed=3423
| random_nat_freq=False
| noise_std=0
| #Parallel process
| max_workers=2

A line starting with the **#** symbol will be considered as a comment.

(Remove the *\|* symbols in the text file.)

.. automodule:: model.parserConfig
   :autosummary:
   :members:
   :undoc-members:
   :show-inheritance:
