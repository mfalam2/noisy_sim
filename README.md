# Simulating noisy quantum circuits with superoperators from IBM devices

extract_noise_model.ipynb shows how to extract superoperators for noisy gates from an IBM device. It should be run on IBM Quantum Lab. A result of executing this file is contained in ibm_perth_gates.pickle

simulator.py is a bare-bones simulator for noisy circuits

noisy_trotter.py implements Trotter evolution (so far, only first order and only the XY model) with noisy superoperators 

unit_tests.ipynb and 10q_runs.py serve as example codes

