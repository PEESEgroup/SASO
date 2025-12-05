# AI for Non-aqueous Electrolyte Design
This software package implements the SCAN (Shaping Conductivity Atlas for Non-aqueous electrolytes) that takes Li-salts, solvents, and conditions to predict the ionic conductivity.

The package provides three major functions:
* Calculate the descriptors based on Li-salts, solvents, and conditions.
* Train a SCAN model with the entire data.
* Predict conductivity based on SCAN model or symbolic regression model.
![web-1](https://github.com/user-attachments/assets/fce81555-1663-422a-81d3-bdf4ca1915da)

## Prerequisites
* torch==2.2.1
* rdkit==2024.3.6
* scikit-learn==1.5.1
* pysr==1.5.2
