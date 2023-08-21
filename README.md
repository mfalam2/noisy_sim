# noisy_sim
Simulates noisy quantum circuits with superoperators from IBM devices

extract_noise_model.ipynb shows how to extract superoperators for noisy gates from an IBM device. It should be run on IBM Quantum Lab. 
simulator.py is a bare-bones simulator for noisy circuits
noisy_trotter.py implement Trotter evolution (so far, only first order and only the XY model) with noisy superoperators 

