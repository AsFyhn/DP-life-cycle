# DP-life-cycle (Term Paper)

This project involves the implementation and simulation of a generalized model, using Python and various scientific libraries. The code performs tasks such as setup of the model described in section **2** in the paper, solving, simulation, estimation, and visualization of the results.

## File Structure

- ``run_all.ipynb``: Jupyter notebook containing the main script to run the project.
- ``model.py``: Defines the gp_model class and related methods to solving the model.
- ``simulations.py``: Contains the Simulator class for simulating model data.
- ``estimation.py``: Implements the SMD class for parameter estimation using simulated method of moments.
- ``plotgenerator.py``: Contains helper functions for plotting.

## Prerequisites

To run the script, ensure you have the following installed:
- Python 3.6 or higher
- Jupyter Notebook
- Required Python libraries: 
    - `numpy`
    - `scipy` 
    - `matplotlib`
    - `pandas`
    - `statsmodels`

## Getting Started

1. Clone the repository to your local machine.
2. Ensure all dependencies are installed.
3. Open the ``run_all.ipynb`` file. 
    - Might need to change the path specified in first code block
    - Set ```python save_figs = True``` if you want to have the figures saved to your machine. Otherwise it will just show.
4. Execute all cells to see the results.
