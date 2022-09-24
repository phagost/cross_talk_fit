'''
Progam to fit the generated noisy data for Solomon equations with
the different time sampling intervals
'''

from models import model_3res

from data import read_data
from capacity import calc_capacity_zeeman_H,\
                         calc_capacity_zeeman_D, \
                         calc_capacity_nz
                         
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

import os
import json

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
    sol = odeint(model_3res, M, time, args=(params,),
              atol=abserr, rtol=relerr)
    return sol

def residual(params, data, exps, only_H=False):
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
    sol = {
        'H' : {},
        'D' : {}
    }
    
    for exp in exps:
    # 11 SOLUTION
        b1 = params[f'b1_{exp}'].value
        b2 = params[f'b2_{exp}'].value
        bnz = params[f'bnz_{exp}'].value
        M = np.array(
            [b1, b2, bnz]
        )
        
        # 
        sol['H'][f'{exp}'] = solution(data[f'H{exp}']['time'], M, params)[:, 0]
        sol['D'][f'{exp}'] = solution(data[f'D{exp}']['time'], M, params)[:, 1]
    
    # Concatenate the result for several fits
    if only_H:
        res = np.concatenate(((sol['H']['11'] - data['H11']['data']) / data['H11']['err'],
                              (sol['H']['01'] - data['H01']['data']) / data['H01']['err']))
    else:
        res = np.concatenate(((sol['H']['11'] - data['H11']['data']) / data['H11']['err'], 
                              (sol['H']['10'] - data['H10']['data']) / data['H10']['err'],
                              (sol['H']['01'] - data['H01']['data']) / data['H01']['err'],
                              (sol['H']['00'] - data['H00']['data']) / data['H00']['err'],
                              (sol['D']['11'] - data['D11']['data']) / data['D11']['err'],
                              (sol['D']['10'] - data['D10']['data']) / data['D10']['err'],  
                              (sol['D']['01'] - data['D01']['data']) / data['D01']['err'], 
                              (sol['D']['00'] - data['D00']['data']) / data['D00']['err']))
    fin = res.flatten()
    return fin

def make_exp_name(exp: str) -> str:
    first = 'max' if exp[0] == '1' else '0'
    sec = 'max' if exp[1] == '1' else '0'
    return f'H$_{{{first}}}$|D$_{{{sec}}}$'

def plot_exp(data, data_fitted, 
             exp: 'str',  ax,  errors: bool=True):
    first = int(exp[0])
    sec = int(exp[1])
    
    color_H = '#ff2b00'
    color_D = '#ff7700'
    
    # Make a number shift
    first = 0 if first == 1 else 1
    sec = 1 if sec == 0 else 0
    
    # H
    ax[first][sec].plot(data[f'H{exp}']['time'], data_fitted[f'H{exp}'], color=color_H, linewidth=5, alpha=0.3, label='H fit', zorder=2)
    ax[first][sec].scatter(data[f'H{exp}']['time'], data[f'H{exp}']['data'], marker='o', facecolor='none', 
                  edgecolor=color_H, s=4, label=f'H')
    if errors:
        ax[first][sec].errorbar(data[f'H{exp}']['time'], data[f'H{exp}']['data'], data[f'H{exp}']['err'], color=color_H, linestyle='None', capsize=1)
    
    # D
    ax[first][sec].plot(data[f'D{exp}']['time'], data_fitted[f'D{exp}'], color=color_D, linewidth=5, alpha=0.3, label='D fit', zorder=2)
    ax[first][sec].scatter(data[f'D{exp}']['time'], data[f'D{exp}']['data'], marker='s', facecolor='none', 
                  edgecolor=color_D, s=4, label=f'D')
    if errors:
        ax[first][sec].errorbar(data[f'D{exp}']['time'], data[f'D{exp}']['data'], data[f'D{exp}']['err'], color=color_D,linestyle='None', capsize=2)
    
    ax[first][sec].plot(data[f'H{exp}']['time'], data_fitted[f'NZ{exp}'], linestyle='dotted', color='green', linewidth=2, label='NZ')
    
    if exp in ('01', '00'):
        ax[first][sec].set_xlabel('time / sec')
    if exp in ('11', '01'):
        ax[first][sec].set_ylabel(r'T$^{-1}$ / K$^{-1}$')
        
    ax[first][sec].title.set_text(make_exp_name(exp))
    ax[first][sec].legend()
    
def _save_report(result, finpath):
    report = fit_report(result)
    filepath = os.path.join(finpath, 'report.txt')
    with open(filepath, "w") as f:
        f.write(report)
    
    filepath = os.path.join(finpath, 'params.json')
    result.params.dump(open(filepath, 'w'), indent=4)

class FitterHD:
    def __init__(self, config):
        self.config = config
        self.exps = {'11', '10', '01', '00'}
        self.nucs = {'H', 'D'}
        self._prepare_fit()
        self.is_fit_performed = False
        
    def my_residual(self, params, data):
        return residual(params, data, exps=self.exps, only_H=self.config["only_H"])
        
    def _count_points(self):
        self.points_H = 0
        self.points_H_only = 0
        self.points_D = 0
        for exp in self.exps:
            if exp in ('11', '01'):
                self.points_H_only += len(self.data[f'H{exp}']['data'])
            self.points_H += len(self.data[f'H{exp}']['data'])
            self.points_D += len(self.data[f'D{exp}']['data'])
            
    def _calc_residuals(self):
        
        if not self.config["only_H"]:
            self.residual_H = np.sum(np.square(self.result.residual[:self.points_H]))
            self.residual_D = np.sum(np.square(self.result.residual[:self.points_D]))
        
        if self.config["only_H"]:
            res = residual(self.result.params, self.data, exps=self.exps, only_H=False)
            self.residual_H = np.sum(np.square(res[:self.points_H]))
            self.residual_D = np.sum(np.square(res[:self.points_D]))
            
            self.residual_H_only = int(np.sum(np.square(self.result.residual)))
    
    def _save_residuals(self):
            
        dict_to_save = {
            'H': {
                'points': self.points_H,
                'residuals': int(self.residual_H)
            },
            'D': {
                'points': self.points_D,
                'residuals': int(self.residual_D)
            }
        }
        
        if self.config["only_H"]:
            dict_to_save['H']['point_H_only'] = self.points_H_only
            dict_to_save['H']['residuals_H_only'] = self.residual_H_only
        
        filepath = os.path.join(self.finpath, 'residuals.json')
        json.dump(dict_to_save, open(filepath, "w"), indent=4)
        
    def _set_sample_composition(self) -> dict:
        """Generate composition filed from the config
        
        Returns:
            dict: composition dict
        """
        composition = {
            'h2o': self.config['h2o_add'],      # ul per 100 ul
            'd2o': 50 - self.config['h2o_add'], 
            'TEMPOL': self.config['conc_tempol'] # mM
        }
        return composition
    
    def _gen_exp_name(self):
        """Generate the name for experiment
        The name in format HD_X-Y[_dt][_only_H] where:
            X - the amount of additional water in DNP juice per 100 ul [ul]
            Y - the TEMPOL concentration [mM]
            optional dt - when the TEMPOL is deuterated
            optional only_H - when only H data is used

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
        exp_name = f'HD_{h2o_add_str}-{TEMPOL}'
        
        if self.config['is_dt']:
            exp_name = exp_name + '_dt'
        
        if self.config["only_H"]:
            exp_name = exp_name + '_onlyH'
            
        return exp_name
    
    def _gen_finpath(self):
        """Generate finpath where reports and graphs will be saved.
        If doesn't folder doesn't exit, than it will be created.
        
        Returns:
            str: the final pathname where all results will be stored 
        """
        start_dir = os.getcwd()
        finpath = os.path.join(start_dir, 'fit_results', self.exp_name)
        
        if not os.path.isdir(finpath):
            os.mkdir(finpath)
            
        return finpath
    
    def _get_init_cond(self):
        # Setting initial value
        init_cond = {
            'H11': self.data['H11']['data'][0],
            'D11': self.data['D11']['data'][0],
            'H10': self.data['H10']['data'][0],
            'D10': 0,
            'H01': 0,
            'D01': self.data['D01']['data'][0],
            'H00': 0,
            'D00': 0
        }
        
        # Setting intial NZ temp for all experiments
        # This is the average temp between D and H reservoirs
        init_cond['NZ'] = ( init_cond['H11'] + init_cond['H10'] 
                          + init_cond['D11'] + init_cond['D01'] ) / 4
        return init_cond
    
    def _set_params(self, params):
        for exp in self.exps:
            params.add(f'b1_{exp}', self.init_cond[f'H{exp}'], vary=False)
            params.add(f'b2_{exp}', self.init_cond[f'D{exp}'], vary=False)
            params.add(f'bnz_{exp}', self.init_cond['NZ'], vary=False)
            
            # Time constants
            tau_1 = 12
            params.add('tau_1', tau_1, min=0.1, max=200, vary=True)
            
            tau_2 = 380
            params.add('tau_2', tau_2, min=10, max=2000, vary=True)
            
            tau_nz = 7
            params.add('tau_nz', tau_nz, min=5e-3, max=1e+2, vary=True)
            
            # Setting heat capacities
            # c_1 depends on the is_dt boolean accounting for the protons from tempol
            if self.config['is_dt']:
                c_1 = calc_capacity_zeeman_H(self.composition['h2o'])
            else:
                c_1 = calc_capacity_zeeman_H(self.composition['h2o'], self.composition['TEMPOL'])
                
            params.add('c_1', c_1, vary=False)
            
            if self.config['is_dt']:
                c_2 = calc_capacity_zeeman_D(self.composition['d2o'], self.composition['TEMPOL'])
            else:
                c_2 = calc_capacity_zeeman_D(self.composition['d2o'])
            params.add('c_2', c_2, vary=False)
            
            c_nz = calc_capacity_nz(self.composition['TEMPOL'])
            params.add('c_nz', c_nz, min=c_nz * 1e-2, max = c_nz * 1e+3, vary=True)
            
    def set_param_values(self, params_config):
        for param_key, param_val in params_config.items():
            self.params[param_key].set(value=param_val)
    
    def _prepare_fit(self):
        # --- FIT PREPARATION ---
        # Set sample composition
        self.composition = self._set_sample_composition()
        # Set experimental name
        self.exp_name = self._gen_exp_name()
        # Read data
        self.data = read_data(self.exp_name, sample_rate=1, 
                              errors=self.config['with_errors'])
        # Save the number of points for H and D
        self._count_points()
        # Make finpath where all data will be saved
        self.finpath = self._gen_finpath()
        # --- END OF FIT PREPARATION ---
        
        # --- FIT ---
        # Set initial conditions
        self.init_cond = self._get_init_cond() 
        # Define the parameters that are going to be optimized
        self.params = Parameters()
        self._set_params(self.params)
            
    
    def make_fit(self, 
                show_plot=True, 
                save_plot=True, 
                print_report=True, 
                save_report=True,
                sample_rate=1):
        # -- RESAMPLING IF NEEDED --- 
        if sample_rate != 1:
            self.data = read_data(self.exp_name, sample_rate=sample_rate, 
                                errors=self.config['with_errors'])
            self._count_points()
        
        # --- FIT ---
        self.result = minimize(self.my_residual, self.params, args=(self.data,), method=self.config['fit_method'])
        self._calc_residuals()
        
        # Calculate data with the result of the best fit
        data_fitted = {}
        
        # 11
        for exp in self.exps:
            ini_exp = [self.init_cond[f'H{exp}'], self.init_cond[f'D{exp}'], self.init_cond['NZ']]
            data_fitted[f'H{exp}'] = solution(self.data[f'H{exp}']['time'], ini_exp, self.result.params)[:, 0]
            data_fitted[f'D{exp}'] = solution(self.data[f'D{exp}']['time'], ini_exp, self.result.params)[:, 1]
            data_fitted[f'NZ{exp}'] = solution(self.data[f'H{exp}']['time'], ini_exp, self.result.params)[:, 2]
        ### --- END OF FIT ---
        
        ### --- REPORT AND PLOTTING ---
        # Print report
        if print_report:
            print(fit_report(self.result))
        # Save report
        if save_report:
            _save_report(self.result, self.finpath)
            self._save_residuals()
        
        # Make plot if requested
        if show_plot or save_plot:
            fig, ax = plt.subplots(2, 2)
            
            for exp in self.exps:
                plot_exp(self.data, data_fitted,
                        exp, ax, errors=self.config['with_errors'])
                
            fig.tight_layout()
        
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

    def emcee(self,
            steps=100,
            progress=True,
            plot=True,
            save_plot=False,
            report=True,
            save_report=False,
            show_walkers=False):
        
        print('--------------------------------------------')
        print('accounting for statistics')
        res = minimize(self.my_residual, method='emcee', nan_policy='omit', burn=30, steps=steps, thin=20,
                        params=self.result.params, args=(self.data,), progress=progress)
        
        if report:
            print('median of posterior probability distribution')
            print('--------------------------------------------')
            print(fit_report(res.params))
            
        if save_report:
            report = fit_report(res.params)
            filepath = os.path.join(self.finpath, 'emcee_report.txt')
            with open(filepath, "w") as f:
                f.write(report)
        
        if show_walkers:
            plt.plot(res.acceptance_fraction, 'o')
            plt.xlabel('walker')
            plt.ylabel('acceptance fraction')
            plt.show()

        if plot or save_plot:
            import corner
            
            fit_vals = [
                res.params.valuesdict()['tau_1'],
                res.params.valuesdict()['tau_2'],
                res.params.valuesdict()['tau_nz'],
                res.params.valuesdict()['c_nz']
            ]

            fig = corner.corner(res.flatchain, 
                        labels=res.var_names,
                        show_titles=False,
                        truths=fit_vals)
            
        if plot:
            plt.show()
            
        if save_plot:    
            filepath = os.path.join(self.finpath, 'emcee_plot')
            
            if os.path.isfile(filepath):
                os.remove(filepath)
            
            fig.savefig(filepath, bbox_inches='tight')

if __name__ == '__main__':
    
    config = {
        'h2o_add': 10,      # ul per 100 ul sample
        'conc_tempol': 60,  # mM
        
        # Set is_dt True if TEMPOL is deuterated
        'is_dt': False,
        
        # Set if only H11 and H01 data should be fitted
        # (as if one wouldn't have D data)
        'only_H': False,
        
        # Set True to account for the errors
        'with_errors': True,

        # SET FIT METHOD
        # 'leastsq' for a local fit
        # 'differential_evolution' for a global fit
        'fit_method': 'differential_evolution'
    }
    
    fitter = FitterHD(config)

    params_config = {
        'tau_1' : 100,
        'tau_2' : 100,
        'tau_nz': 100
    }
    
    fitter.set_param_values(params_config)
    
    fitter.make_fit(show_plot=True,
                save_plot=True,
                print_report=True,
                save_report=False,
                sample_rate=1)

    # fitter.emcee(steps=1000,
    #             progress=True,
    #             plot=True,
    #             save_plot=False,
    #             report=True,
    #             save_report=False)
    