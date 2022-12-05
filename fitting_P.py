'''
Progam to fit the generated noisy data for Solomon equations with
the different time sampling intervals
'''

from models import m_3res_f_cnz_X

from data import read_data
from capacity import calc_capacity_zeeman_H,\
                         calc_capacity_zeeman_P, \
                         calc_capacity_zeeman_D
                         
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import os

from lmfit import minimize, Parameters, fit_report

def solution(time, M, params):
    """Find the model solution for a specific set of time domain, 
    initial condition, parameters
    
    Args:
        time (ndarray, dtype=float): time domain
        M (ndarray, dtype=float): initial values
        params (dict): dict of the given parameters
    """
    abserr = 1.0e-8
    relerr = 1.0e-6
    sol = odeint(m_3res_f_cnz_X, M, time, args=(params,),
              atol=abserr, rtol=relerr)
    return sol

def residual(params, data, exps):
    """Returns residuals for data obtained with the different time sampling.

    Args:
        params (dict of Parameter): inital condition and parameters.
            Initial conditions are assigned to individual variable right at the head
            of the function.
        time (dict): time list for different experiments
            I - for I magnetization evolution
            S - for S magnetization evolution
        data (list): list of data for different experiments

    Returns:
        ndarray: _description_
    """
    sol = {}
    
    for exp in exps:
        b1 = params[f'b1_{exp}'].value
        b2 = params[f'b2_{exp}'].value
        b3 = params[f'b3_{exp}'].value
        bnz = params[f'bnz_{exp}'].value
        M = np.array(
            [b1, b2, b3, bnz]
        )
        
        # 
        sol[f'{exp}'] = solution(data[f'P{exp}']['time'], M, params)[:, 2]
    
    # Concatenate the result for several fits
    res = np.concatenate((
        (sol['1'] - data['P1']['data']) / data['P1']['err'], 
        (sol['0'] - data['P0']['data']) / data['P0']['err']
        ))
            
    fin = res.flatten()
    return fin

def make_exp_name(exp: str) -> str:
    if exp == '1':
        exp = 'max'
    return f'P$_{{{exp}}}$'

def plot_exp(data, data_fitted, 
             exp: 'str',  ax,  errors: bool=True):
    
    num = int(exp[0])
    
    # Change the order of numbers
    num = 1 if num == 0 else 0
    
    # exp_name = make_exp_name(exp)   
    ax[num].scatter(data[f'P{exp}']['time'], data[f'P{exp}']['data'], marker='o', facecolor='none', 
                  edgecolor='#05d5ff', s=4, label='P')
    
    ax[num].plot(data[f'P{exp}']['time'], data_fitted[f'P{exp}'], color='#05d5ff', linewidth=10, alpha=0.3, label='fit')
    
    
    ax[num].set_xlabel('time / sec')
    if num == 0:
        ax[num].set_ylabel(r'$\beta$ / K$^{-1}$')
    ax[num].title.set_text(make_exp_name(exp))
    ax[num].legend()
    
    if errors:
        ax[num].errorbar(data[f'P{exp}']['time'], data[f'P{exp}']['data'], data[f'P{exp}']['err'], color='#05d5ff', linestyle='None', capsize=1)
    
    
def _save_report(result, finpath):
    report = fit_report(result)
    filepath = os.path.join(finpath, 'report_X.txt')
    with open(filepath, "w") as f:
        f.write(report)
    
    filepath = os.path.join(finpath, 'params.json')
    result.params.dump(open(filepath, 'w'), indent=4)

class FitterX:
    def __init__(self, config, load_params=True):
        self.config = config
        self.exps = {'1', '0'}
        self.nucs = {'P'}
        self.is_fit_performed = False
        
        # --- FIT PREPARATION ---
        # Set sample composition
        self.composition = self._set_sample_composition()
        # Set experimental name
        self.exp_name = self._gen_exp_name()
        # Read data
        self.data = read_data(self.exp_name, errors=self.config['with_errors'])
        # Make finpath where all data will be saved
        self.finpath = self._gen_finpath()
        # --- END OF FIT PREPARATION ---
        
        # --- FIT ---
        # Set initial conditions
        self.init_cond = self._get_init_cond()
        
        # Define the parameters that are going to be optimized
        self.params = Parameters()
        self._set_params(load_params=load_params)
        
    def my_residual(self, params, data):
        return residual(params, data, exps=self.exps)
    
    
    def _set_params(self, load_params=True):
        # Set parameters specfic for HD DNP juice
        if load_params:
            self._set_loaded_parameters()
        else:
            self._set_manual_parameters()

        # Set parameters specific for P experiments
        # ---
        # P experiment initial conditions
        for exp in self.exps:
            self.params.add(f'b1_{exp}', self.init_cond[f'NZ'], vary=False)
            self.params.add(f'b2_{exp}', self.init_cond[f'NZ'], vary=False)
            self.params.add(f'b3_{exp}', self.init_cond[f'P{exp}'], vary=False)
            self.params.add(f'bnz_{exp}', self.init_cond['NZ'], vary=False)
        
        # P experiments kinetic parameters
        tau_3 = 1000
        self.params.add('tau_3', tau_3, min=10, max=3000, vary=True)
        
        c_3 = calc_capacity_zeeman_P(self.composition['k2hpo4'])
        self.params.add('c_3', c_3, vary=False)
    
    def _set_loaded_parameters(self):
        
        self.params.load(
            open(os.path.join(self.finpath.replace('P', 'HD'), 'params.json'), 'r')
        )
        
        HD_exps = ('11', '10', '01', '00')
        for exp in HD_exps:
            del self.params[f'b1_{exp}']
            del self.params[f'b2_{exp}']
            del self.params[f'bnz_{exp}']
        
        self.params['tau_1'].vary = False
        self.params['tau_2'].vary = False
        self.params['tau_nz'].vary = False
        self.params['f1'].vary = False
        self.params['c_nz'].vary = False
    
    def set_par(self, param_key, value, vary=False):
        self.params[param_key].set(value=value)
        self.params[param_key].set(vary=vary)
        
    def _set_manual_parameters(self):
        # SET TIME CONSTANTS
        tau_1 = 4.77464535
        self.params.add('tau_1', tau_1, min=0.1, max=200, vary=False)
        
        tau_2 = 248.794421
        self.params.add('tau_2', tau_2, min=10, max=2000, vary=False)
        
        tau_nz = 4.39540880
        self.params.add('tau_nz', tau_nz, min=5e-3, max=1e+2, vary=False)
        
        # SET HEAT CAPACITIES
        c_1 = calc_capacity_zeeman_H(self.composition['h2o'], self.composition['TEMPOL'])
        self.params.add('c_1', c_1, vary=False)
        
        c_2 = calc_capacity_zeeman_D(self.composition['d2o'])
        self.params.add('c_2', c_2, vary=False)
        
        c_nz = 7.6386e+38
        self.params.add('c_nz', c_nz, vary=False)
    
    def _get_init_cond(self):
        init_cond = {
            'P1': self.data['P1']['data'][0],
            'P0': 0
        }
        init_cond['NZ'] = init_cond['P1']
        return init_cond
    
    def _set_sample_composition(self) -> dict:
        """Generate composition filed from the config
        
        Returns:
            dict: composition dict
        """
        composition = {
            'h2o': self.config['h2o_add'],      # ul per 100 ul
            'd2o': 50 - self.config['h2o_add'], 
            'k2hpo4': self.config['k2hpo4'],     # M
            'TEMPOL': self.config['conc_tempol'] # mM
        }
        return composition
    
    def _gen_exp_name(self):
        """Generate the name for experiment
        The name in format P_X-Y where:
            X - the amount of additional water in DNP juice per 100 ul [ul]
            Y - the TEMPOL concentration [mM]

        Args:
            self.composition field should exist

        Returns:
            str: experiment name 
        """
        # Strange way to find the amount of additional water
        if not isinstance(self.composition['h2o'], int):
            first = str(int(self.composition['h2o']))
            second = str(int((self.composition['h2o'] * 10) % 10))
            h2o_add_str = first + '&' + second
        else:
            h2o_add_str = str(self.composition['h2o'])
        
        
        TEMPOL = self.composition['TEMPOL']
        exp_name = f'P_{h2o_add_str}-{TEMPOL}'
        return exp_name
    
    def _gen_finpath(self):
        """Generate finpath where reports and graphs will be saved.
        If doesn't folder doesn't exit, than it will be created.
        
        Returns:
            str: the final pathname where all results will be stored 
        """
        start_dir = os.getcwd()
        
        expname =  self.exp_name
        err_name = "err" if self.config["with_errors"] else "noerr"
        finpath = os.path.join(start_dir, 'fit_results', self.config["model"], 
                               err_name, expname)
        
        if not os.path.isdir(finpath):
            os.makedirs(finpath, exist_ok=True)
            
        return finpath
    
    def _make_plot(self, data_fitted):
        
        font = {'family' : 'Helvetica',
                'size'   : 18}

        matplotlib.rc('font', **font)

        # plt.rc('font', family='Helvetica')
        fig, ax = plt.subplots(1, 2)
            
        for exp in self.exps:
            plot_exp(self.data, data_fitted,
                    exp, ax, errors=self.config['with_errors'])
        
        fig.set_size_inches(10, 4.5)
        
        return fig, ax
    
    def make_fit(self,
                 load_params=True,
                 show_plot=True, 
                 save_plot=True, 
                 print_report=True, 
                 save_report=True):
        
        self.result = minimize(self.my_residual, self.params, args=(self.data,), 
                               method=self.config['fit_method'])
        
        # Calculate data with the result of the best fit
        data_fitted = {}
        
        for exp in self.exps:
            ini_exp = [self.init_cond[f'NZ'], self.init_cond[f'NZ'], self.init_cond[f'P{exp}'], self.init_cond['NZ']]
            data_fitted[f'P{exp}'] = solution(self.data[f'P{exp}']['time'], ini_exp, self.result.params)[:, 2]
        ### --- END OF FIT ---
        
        ### --- REPORT AND PLOTTING ---
        # Print report
        if print_report:
            print(fit_report(self.result))
        # Save report
        if save_report:
            _save_report(self.result, self.finpath)
        
        # Make plot if requested
        if show_plot or save_plot:
            fig, ax = self._make_plot(data_fitted)
                
        # Show plot if requested
        if show_plot:
            plt.show()
        # Save plot if requested
        if save_plot:
            filepath = os.path.join(self.finpath, 'fit_plot')
            
            # check if the plot is already exist
            # if yes, it should be deleted
            if os.path.isfile(filepath):
                os.remove(filepath)
            
            fig.savefig(filepath, bbox_inches='tight')     
        ### --- END OF REPORT AND PLOTTING ---

if __name__ == '__main__':
    
    config = {
        'h2o_add': 10,      # ul per 100 ul sample
        'conc_tempol': 70,  # mM
        'k2hpo4': 0.5,      # M
        
        # Set True to account for the errors
        'with_errors': True,
        
        # Set the model method
        # All model names start with m_
        "model": "m_3res_f_cnz",
        
        # SET FIT METHOD
        # 'leastsq' for a local fit
        # 'differential_evolution' for a global fit
        # 'fit_method': 'differential_evolution'
        'fit_method': 'leastsq'
    }
    
    fitter = FitterX(config)
    
    fitter.set_par('tau_3', value=877.0, vary=True)
    
    fitter.make_fit(show_plot=True,
                    save_plot=True,
                    print_report=True,
                    save_report=True)
    