"""
Different methods to calculate heat capacities based no my sample composition.

There are four capacities I usually want to calculate:
    1H, 2H, 31P zeeman
    NZ capacities, that may be approximated by DD capacity (but maybe not so strict)
    
These values should be calculated knowing the value of sample compositions. These values
are defined as:
    P: concentration 
    H: volume part of H2O
    D: volume part of D2O
    
Some nuclei have additional sourse:
    H: impurities from D2O and glycerol-d8
    D: glycerol-d8
"""

import spindata as sd   # for gammas
import numpy as np
import scipy.constants as spc

def calc_capacity_zeeman_H(vol_add_h2o: float, conc_tempol: float=0, info: bool=False) -> float:
    """Caculate Zeeman heat capacity for 1H given the amount of additional water.
    (TODO) The protons coming from glycerol-d8 is also included into calculation.

    Args:
        additional_water (float): the amount of additional water in the samle composition
            normilized per 100ul whole sample volume, ul.
            In my experiments it may vary from 0 to 25.
            
    Returns:
        float: Zeeman heat hapacity of the 1H
    """
    # Setting some constants, such as
    #   d_h2o   : h2o density
    #   mw_h2o  : h2o molar weight
    #   v_fin   : the final sample volume 
    d_h2o = 1           # [g ml-1] , the same as [g cm-3]
    mw_h2o = 18         # [g mol-1]
    vol_fin = 100       # ul
        
    # The final formula for cm-3 is n_h2o = N_A * d_h2o * vol_add / (mw * vol_fin)
    n_h2o = spc.N_A * d_h2o * vol_add_h2o / (mw_h2o * vol_fin)    
    
    # The amount of protons is twice the amount of water
    n_h_from_h2o = 2 * n_h2o
    
    # Correction from glycerol
    n_h_from_glycerol = glycerol_corr_h()
    
    # Correction from TEMPOL
    n_h_from_tempol = 18*cacl_conc_tempol(conc_tempol)
    
    # Correcting for the amount of protons in the glycerol
    # And the amount of protons in the TEMPOL (there are 18! H in the tempol)
    n_h = n_h_from_h2o + n_h_from_glycerol + n_h_from_tempol
    
    
    if info:
        print(f"H from h2o {n_h_from_h2o:.2e}\n"
              f"H from glycerol {n_h_from_glycerol:.2e}\n"
              f"H from tempol {n_h_from_tempol:.2e}\n"
              f"H overall {n_h:.2e}")
    
    return heat_capacity_zeeman(B = 6.7, 
                                n = n_h, 
                                I = 1/2,
                                nuclei='1H')
    
def calc_capacity_zeeman_P(conc_p: float) -> float:
    """Caculate Zeeman heat capacity for 31P given the concentration of K2HPO4.

    Args:
        conc_p (float): the K2HPO4 concentration in [M].
            In my experiments it is set to be 0.5.
            
    Returns:
        float: Zeeman heat hapacity of the 31P
    """
    
    # The formula is just N_A * conc_p
    # The factor 10**-3 is to transfer L-1 to ml-1 (cm-3 respectively)
    n_p = spc.N_A * conc_p * 1e-3
        
    return heat_capacity_zeeman(B = 6.7, 
                                n = n_p, 
                                I = 1/2,
                                nuclei='31P')
    
def calc_capacity_zeeman_D(vol_add_d2o: float, conc_tempol: float=0, info: bool=False) -> float:
    """Caculate Zeeman heat capacity for 2H (D) given the amount of additional d2o.
    The d2o coming from glycerol-d8 is aslo calculated.

    Args:
        vol_add_d2o (float): the amount of additional d2o in the samle composition
            normilized per 100ul whole sample volume, ul.
            
    Returns:
        float: Zeeman heat hapacity of the 2H (D)
    """
    
    # Setting some constants, such as
    #   d_d2o   : d2o density
    #   mw_d2o  : d2o molar weight
    #   v_fin   : the final sample volume 
    d_d2o = 1.11       # [g ml-1] , the same as [g cm-3]
    mw_d2o = 29     # [g mol-1]
    vol_fin = 100     # ul
    
    # The final formula for cm-3 is n_d2o = N_A * d_d2o * vol_add / (mw * vol_fin)
    n_d2o = spc.N_A * d_d2o * vol_add_d2o / (mw_d2o * vol_fin)    
    
    # The amount of protons is twice the amount of water
    n_d_from_d2o = 2 * n_d2o
    
    # Correction from glycerol
    n_d_from_glycerol = glycerol_corr_d()
    
    # Correction from TEMPOL
    n_d_from_tempol = 18*cacl_conc_tempol(conc_tempol)
    
    # Finally, let's add deuterium from glycerol
    n_d = n_d_from_d2o + n_d_from_glycerol + n_d_from_tempol
    
    if info:
        print(f"D from d2o {n_d_from_d2o:.2e}\n"
              f"D from glycerol {n_d_from_glycerol:.2e}\n"
              f"D from tempol {n_d_from_tempol:.2e}\n"
              f"D overall {n_d:.2e}")
    
    
    
    return heat_capacity_zeeman(B = 6.7, 
                                n = n_d, 
                                I = 1,
                                nuclei='2H')
    
def calc_capacity_nz(conc_tempol, delta=3.06*5.25e-3, B=6.7):
    """Estimate heat capacity of the nz reservoir as dipolar reservoir.
    The approximate value for TEMPOL was taken from here 10.1103/PhysRevB.74.134418.
    The 3.06 correction correpsond to the fact, that our linewidth should be about 450 MHz

    Args:
        conc_tempol (float): TEMPOL concentration in mM
        delta (float, optional): FWHM of TEMPOL spectra, in T. Defaults to 5.25e-3.

    Returns:
        float: heat capacity for dipolar reservoir
    """
    # recalculate delta from T to rad s-1
    delta = sd.gamma('E') * delta
    
    # recalculate concentration from mM to cm-3
    conc_tempol = cacl_conc_tempol(conc_tempol)
    
    return heat_capacity_nz(delta, conc_tempol)

def cacl_conc_tempol(conc_tempol):
    """Calculate the cm-3 concentration of the tempol

    Args:
        conc_tempol (double): concentration of the tempol in mM

    Returns:
        double: concentration of the tempol in cm-3
    """
    # recalculate concentration from mM to cm-3
    conc_tempol = spc.N_A * conc_tempol * 1e-3 * 1e-3

    return conc_tempol

def heat_capacity_zeeman(B, n, I=1/2, nuclei='1H') -> float:
    """Calculate the Zeeman heat capacity.

    Args:
        B (float): magnetic field in Tesla [T]
        n (float): spin species concentration [sec**-2 cm**-3]
        I (float, optional): the nuclei momentum. Defaults to 1/2.
        nuclei (str, optional): nuclei species. Defaults to '1H'.
            For my experiments, it may be either '1H', '2H' or '31P'
    Returns:
        float: heat capacity of Zeeman reservoir in energy**2 * n units
    """
    gamma = sd.gamma(nuclei)
    omega = gamma * B
    return (I*(I+1)/3) * n * omega**2
        
def heat_capacity_nz(delta, n, I = 1/2):
    """Calculate electron DD heat capacity as nz heat capacity.

    Args:
        delta (double): the width of EPR spectrum in rad s-1
        n (double): electron species concentration [sec**-2 cm**-3]
        I (double, optional): electron spin value. Defaults to 1/2.

    Returns:
        foat: heat capasity of electron DD reservoir in energy**2 * n units
    """
    return (I*(I+1)/3) * n * delta**2

def glycerol_corr_d(vol_gly=50) -> float:
    """Calculate the correction to the 2H(D) deuterium concentration coming from glycerol-d8

    Args:
        vol_gly (int, optional): glycerol value in [ul] per 100ul sample. Defaults to 50.

    Returns:
        float: float: concentration of 2H (D) coming from glycerol in [cm**3]
    """
    # Setting some constants
    d_gly = 1.36
    mw_gly = 101
    vol_fin = 100
    
    # The final formula for cm-3 is n_d2o = N_A * d_d2o * vol_add / (mw * vol_fin)
    n_gly = spc.N_A * d_gly * vol_gly / (mw_gly * vol_fin)
    
    # Glycerol is 0.98 D purity
    return 8 * n_gly * 0.98


def glycerol_corr_h(vol_gly=50) -> float:
    """Calculate the correction to the 2H(D) deuterium concentration coming from glycerol-d8

    Args:
        vol_gly (int, optional): glycerol value in [ul] per 100ul sample. Defaults to 50.

    Returns:
        float: float: concentration of 2H (D) coming from glycerol in [cm**3]
    """
    # Setting some constants
    d_gly = 1.36
    mw_gly = 101
    vol_fin = 100
    
    # The final formula for cm-3 is n_d2o = N_A * d_d2o * vol_add / (mw * vol_fin)
    n_gly = spc.N_A * d_gly * vol_gly / (mw_gly * vol_fin)
    
    # Glycerol is 0.98 D purity
    return 8 * n_gly * 0.02


def sample_info(composition: dict, info=False):
    
    # Set the composition
    h2o = composition['h2o']
    d2o = composition['d2o']
    k2hpo4 = composition['k2hpo4']
    tempol = composition['TEMPOL']
    
    print('------------------')
    # Evaluate DNP juice params
    if composition['is_dt']:
        c_h = calc_capacity_zeeman_H(h2o, info=info)
    else:
        c_h = calc_capacity_zeeman_H(h2o, tempol, info=info)
    print(f'The zeeman capacity of 1H for {h2o} ul additional '\
          f'water is {c_h:.2e}')
    
    print('---')
    
    if composition['is_dt']:
        c_d = calc_capacity_zeeman_D(d2o, tempol, info=info)
    else:
        c_d = calc_capacity_zeeman_D(d2o, info=info)
    print(f'The zeeman capacity for 2H (D) for {d2o} ul additional '\
          f'd2o is {c_d:.2e}')
    
    print('---')
    
    print(f'The zeeman capacity for 31P for {k2hpo4} M '\
          f'k2hpo4 is {calc_capacity_zeeman_P(k2hpo4):.2e}')
    
    print('---')
    
    print(f'The non-zeeman capacity for e for {tempol} mM concentration '\
    f'is {calc_capacity_nz(tempol):.2e}')
    print('------------------')
    

if __name__ == '__main__':
    """Calculate heat capacities for 10/40/50 h2o/d2o/glycerol-d8 
    """
    composition = {
        'h2o': 10,         # ul per 100 ul
        'k2hpo4': 0.5,      # M
        'TEMPOL': 50,       # mM
        'is_dt': False     # if TEMPOL is deuterated
    }
    
    composition['d2o'] = 50 - composition['h2o']
    
    sample_info(composition, info=True)
    