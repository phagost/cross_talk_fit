# Cross talk data and fitting procedure

This repository contains the processed data for HD and P cross-talk experiments, and also python routines to process them.

## Data

The data is contained in the data folder in the format `HD_X_Y[_dt]` or `P_X_Y`, where for 100ul DNPsample with 50% volume percentage glycerol-d8:

* `HD` or `P` - the type if nucei that will be fitted
* `X` - amount of additional H<sub>2</sub>O in ul
* `Y` - the concentration of TEMPOL in mM
* optional `[dt]` - if the TEMPOL is deuterated

In each folder `HD` folder, there 8 `.dat` files with naming covention `NucN1N2`:

* `Nuc` - H or D, detection nuclei
* `N1` - 1 or 0, the inital state of H nuclei, where 1 corresponds for maximum polarization and 0 correspons to depolorized state
* `N2` - 1 or 0, the same for D nuclei

In `P` folders there naming `PN`, where `N` is either 1 or 0 and corresponds to polarization states of P.

Each data set has three columns: time, **inverse temperature** (preprocessing procedure is not given here) and the error

## Python routines
There are two types of python files : main python fitters and helpers with different functionality. To use them, some non-conventional packages must be installed:

* lmfit
* corner
* numdifftools
* emcee
* spindata
* tqdm

### Main fitters
`fitting_HD.py` and `fitting_P` for fitting HD and P data, respectively. The example of thier usage is shown in the `fitting_example.ipynb` notebook.
### Helpers

* `capacity.py` - calculated heat capacity from sample composition
* `data.py` - data reader
* `models.py` - file with different models that were tried during the course of the work
