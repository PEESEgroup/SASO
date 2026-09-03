# AI for Salt-Solvent Formulation Generation
This software package implements the SASO that generates salt-solvent formulations with high ionic conductivity.
<img width="1056" height="463" alt="image" src="https://github.com/user-attachments/assets/a01f1cfa-4d99-478c-a543-28a6ff6dccba" />


The package provides three major functions:
* Generate formulations (Li-salts, solvents, and conditions) with target ionic conductivity.
* Predict ionic conductivity using the surrogate model (SCAN model).
* Construct the MD-simulation box based on the formulations.


## Prerequisites
* torch==2.4.1
* rdkit==2024.3.6
* numpy==1.16.4
* pandas==2.2.2

The easiest way of installing the prerequisites is via `conda`. After installing `conda`, run the following command to create a new environment named `saso` and install all prerequisites:

    conda upgrade conda
    conda create -n saso python=3.12 scikit-learn pytorch rdkit pysr

This creates a `conda` environment for running `SASO`. Before using `SASO`, activate the environment by:
    
    source activate saso

Alternatively, `environment.yaml` provides the dependencies for creating running environment. Then, in directory `model`, you can test if all the prerequisites are installed properly by running:

    python train.py

After you finished using `SASO`, exit the environment by:

    source deactivate

## Models
We provide the model files in `model` directory.
* `model`: MLPD, CVAE, R-MLPD, and R-CVAE models were implemented for generating the salt-solvent formulations, to run the model:

        python train.py

* `trained_pt`: trained model weights of MLPD, CVAE, R-MLPD, R-CVAE models

* `trained_pth`: trained model weights of SCAN models

## Scripts
We provide the practical scripts in `examples` to predict the ionic conductivity based on our well-trained SCAN models, and in `utils` directory for calculating the ionic conductivity and constructing simulation box.
* **Ionic conductivity prediction**

You can predict ionic conductivity based on our SCAN models, by running:

        python predict.py
  
* **Simulation box construction**

This tool is useful to calculate the number of molecules for Li-salts and two solvents in the simulation box, according to parameters: `box_size`, `density`, `salt_concentration`, `salt_molar_mass`, `solvent1_molar_mass`,` solvent2_molar_mass`, `solvent_mass_ratio`, by running:

        python box_construction.py

* **Conductivity calculation**

After obtaining the diffusion coefficent from the MD simultions, you can use this tool to calculate the ionic conductivity based on Arrhenius equation:

        python conductivity_calculation.py


## Data
To reproduce our paper, you can download the corresponding datasets in `data` directory.
* `training data`: lists the compiled data, including k values, temperature, concentration/unti, salt, solvent. The temperature was scaled by a factor of 100.
* `salt_MO.txt`: HOMO-LUMO pairs of salt molecules based on QM calculations.
* `solvent_MO.txt`: HOMO-LUMO pairs of solvent molecules based on QM calculations.
* `designed_formulation.csv`: designed salt-solvent formulation based on chemical insights, whose ionic conductivities were predicted using SCAN model.

## Author contributions
This software was primarily written by `Dr. Zhilong Wang` who is advised by `Prof. Fengqi You`.


## How to cite
Please cite the following work if you want to use SASO:

    Zhilong Wang, Wentao Hou, Xianyong Wu*, Fengqi You*. Submitted, (2026)
