'''
Progam to fit the generated noisy data for Solomon equations with
the different time sampling intervals
'''
import importlib
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import os
import json

from lmfit import minimize, Parameters, fit_report

from data import read_data
from capacity import calc_capacity_zeeman_H,\
                         calc_capacity_zeeman_D, \
                         calc_capacity_nz
                         
MODELS_3_RES = (
    "m_3res", 
    "m_3res_f",
    "m_3res_2f",
    "m_3res_f_cnz",
)

MODELS_4_RES = (
    "m_4res_f_cnz",
    "m_4res_2f",
    "m_4res_2f_cnz",
    "m_4res_3f",
)
                         
def make_exp_name(exp: str) -> str:
    first = 'm' if exp[0] == '1' else '0'
    sec = 'm' if exp[1] == '1' else '0'
    return f'H$_{{{first}}}$D$_{{{sec}}}$'

def plot_exp(data, data_fitted, 
             exp: 'str',  ax,  errors: bool=True, config=None):
    first = int(exp[0])
    sec = int(exp[1])
    
    color_H = '#ff2b00'
    color_D = '#ff7700'
    
    # Make a number shift
    first = 0 if first == 1 else 1
    sec = 1 if sec == 0 else 0
    
    ax_current = ax[first][sec]
    
    # H
    ax_current.plot(data[f'H{exp}']['time'], data_fitted[f'H{exp}'], color=color_H, linewidth=5, alpha=0.3, label='H fit', zorder=2)
    ax_current.scatter(data[f'H{exp}']['time'], data[f'H{exp}']['data'], marker='o', facecolor='none', 
                  edgecolor=color_H, s=4, label=f'H')
    if errors:
        ax_current.errorbar(data[f'H{exp}']['time'], data[f'H{exp}']['data'], data[f'H{exp}']['err'], color=color_H, linestyle='None', capsize=1)
    
    # D
    ax_current.plot(data[f'D{exp}']['time'], data_fitted[f'D{exp}'], color=color_D, linewidth=5, alpha=0.3, label='D fit', zorder=2)
    ax_current.scatter(data[f'D{exp}']['time'], data[f'D{exp}']['data'], marker='s', facecolor='none', 
                  edgecolor=color_D, s=4, label=f'D')
    if errors:
        ax_current.errorbar(data[f'D{exp}']['time'], data[f'D{exp}']['data'], data[f'D{exp}']['err'], color=color_D,linestyle='None', capsize=2)
    
    ax_current.plot(data[f'H{exp}']['time'], data_fitted[f'NZ{exp}'], linestyle='dotted', color='green', linewidth=2, label='NZ')
    
    if exp in ('01', '00'):
        ax_current.set_xlabel('time / sec')
    if exp in ('11', '01'):
        ax_current.set_ylabel(r'$\beta$ / K$^{-1}$')
        
    if exp in ("01", "00", "10"):
        axins = ax_current.inset_axes([0.4, 0.4, 0.57, 0.57])
        
        axins.plot(data[f'H{exp}']['time'], data_fitted[f'H{exp}'], color=color_H, linewidth=5, alpha=0.3, label='H fit', zorder=2)
        axins.scatter(data[f'H{exp}']['time'], data[f'H{exp}']['data'], marker='o', facecolor='none', 
                  edgecolor=color_H, s=4, label=f'H')
        if errors:
            axins.errorbar(data[f'H{exp}']['time'], data[f'H{exp}']['data'], data[f'H{exp}']['err'], color=color_H, linestyle='None', capsize=1)
        
        axins.plot(data[f'H{exp}']['time'], data_fitted[f'NZ{exp}'], linestyle='dotted', color='green', linewidth=2, label='NZ')
        
        axins.plot(data[f'D{exp}']['time'], data_fitted[f'D{exp}'], color=color_D, linewidth=5, alpha=0.3, label='D fit', zorder=2)
        axins.scatter(data[f'D{exp}']['time'], data[f'D{exp}']['data'], marker='s', facecolor='none', 
                  edgecolor=color_D, s=4, label=f'D')
        
        if errors:
            ax_current.errorbar(data[f'D{exp}']['time'], data[f'D{exp}']['data'], data[f'D{exp}']['err'], color=color_D,linestyle='None', capsize=2)
        
        # sub region of the original image
        if exp in ("01", "00"):
            if not config:
                x1, x2, y1, y2 = -2, 15, -0.2, 1.7
            else:
                if config["is_dt"]:
                    # x1, x2, y1, y2 = -2, 15, -0.2, 1.7
                    x1, x2, y1, y2 = -2, 25, -0.2, 2
                else:
                    x1, x2, y1, y2 = -2, 40, -0.2, 4
        if exp in ("10"):
            if not config:
                x1, x2, y1, y2 = -2, 500, -0.2, 4
            else:
                if config["is_dt"]:
                    x1, x2, y1, y2 = -2, 500, -0.2, 4
                else:
                    x1, x2, y1, y2 = -2, 500, -0.2, 4
            
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        # axins.set_xticklabels([])
        # axins.set_yticklabels([])
        
        ax_current.indicate_inset_zoom(axins, edgecolor="black")
    
    
    # Rewrite x ticks
    start, end = ax_current.get_xlim()
    ax_current.xaxis.set_ticks(np.arange(0, end, 300))
    
    # Set xlims
    ax_current.set_xlim(-50, 1000)
    
    ax_current.title.set_text(make_exp_name(exp))
    if exp in ("11",):
        ax_current.legend()
    
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
        self._prepare_fitter()
        self.is_fit_performed = False
        try:
            self.model = getattr(importlib.import_module("models"), config['model'])
        except:
            print("!!! Wrong model name !!!")
        self.data_fitted = self.make_prediction(fitted=False)
            
    def solution(self, time, M, params):
        """Find the model solution for a specific set of time domain, 
        initial condition, parameters
        
        Args:
            time (ndarray, dtype=float): time domain
            M (ndarray, dtype=float): initial values
            params (dict): dict of the given parameters
        """
        abserr = 1.0e-8
        relerr = 1.0e-6
        sol = odeint(self.model, M, time, args=(params,),
                atol=abserr, rtol=relerr)
        return sol
    
    def residual(self, params, data, exps, only_H=False):
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
            if self.config["model"] in MODELS_3_RES:
                b1 = params[f'b1_{exp}'].value
                b2 = params[f'b2_{exp}'].value
                bnz = params[f'bnz_{exp}'].value
                M = np.array(
                    [b1, b2, bnz]
                )
                
            if self.config["model"] in MODELS_4_RES:
                b1 = params[f'b1_{exp}'].value
                b2 = params[f'b2_{exp}'].value
                bnz = params[f'bnz_{exp}'].value
                # Hidden bulk is about the same as nz
                b1_hb = params[f'bnz_{exp}'].value
                M = np.array(
                    [b1, b2, bnz, b1_hb]
                )
            
            # 
            sol['H'][f'{exp}'] = self.solution(data[f'H{exp}']['time'], M, params)[:, 0]
            sol['D'][f'{exp}'] = self.solution(data[f'D{exp}']['time'], M, params)[:, 1]
        
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
    
    def my_residual(self, params, data):
        return self.residual(params, data, exps=self.exps, only_H=self.config["only_H"])
        
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
            res = self.residual(self.result.params, self.data, exps=self.exps, only_H=False)
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
            'd2o': (50 - self.config['h2o_add']), 
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
            
        return exp_name
    
    def _gen_finpath(self):
        """Generate finpath where reports and graphs will be saved.
        If doesn't folder doesn't exit, than it will be created.
        
        Returns:
            str: the final pathname where all results will be stored 
        """
        start_dir = os.getcwd()
        
        expname = (self.exp_name + '_onlyH') if self.config['only_H'] else self.exp_name
        err_name = "err" if self.config["with_errors"] else "noerr"
        finpath = os.path.join(start_dir, 'fit_results', self.config["model"], 
                               err_name, expname)
        
        if not os.path.isdir(finpath):
            os.makedirs(finpath, exist_ok=True)
            
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
            params.add('tau_1', tau_1, min=0.1, max=100, vary=True)
            
            tau_2 = 380
            params.add('tau_2', tau_2, min=50, max=600, vary=True)
            
            tau_nz = 7
            params.add('tau_nz', tau_nz, min=5e-1, max=4e+1, vary=True)
            
            # Setting heat capacities
            # c_1 depends on the is_dt boolean accounting for the protons from tempol
            if self.config['is_dt']:
                c_1 = calc_capacity_zeeman_H(self.composition['h2o'])
            else:
                c_1 = calc_capacity_zeeman_H(self.composition['h2o'], self.composition['TEMPOL'])
                
            params.add('c_1', c_1, min=c_1 * 1e-1, max = c_1 * 10, vary=False)
            
            if self.config['is_dt']:
                c_2 = calc_capacity_zeeman_D(self.composition['d2o'], self.composition['TEMPOL'])
            else:
                c_2 = calc_capacity_zeeman_D(self.composition['d2o'])
            params.add('c_2', c_2, min=c_2 * 1e-1, max = c_2 * 10, vary=False)
            
            if self.config['model'] in ("m_3res", "m_3res_f_cnz", "m_4res_f_cnz", "m_4res_2f_cnz"):
                c_nz = calc_capacity_nz(self.composition['TEMPOL'])
                if self.config['model'] in ("m_3res",):
                    params.add('c_nz', c_nz, min=c_nz, max = c_nz * 1e+3, vary=True)
                if self.config['model'] in ("m_3res_f_cnz", "m_4res_f_cnz", "m_4res_2f_cnz"):
                    params.add('c_nz', c_nz, min=c_nz, max = c_nz * 1e+3, vary=False)
            
            if self.config["model"] in ("m_3res_f", "m_3res_2f", "m_3res_f_cnz", "m_4res_f_cnz", "m_4res_2f", "m_4res_3f", "m_4res_2f_cnz"):
                f1 = 0.1
                params.add('f1', f1, min=0.01, max=0.4, vary=True)
            
            if self.config["model"] in ("m_3res_2f", "m_4res_2f", "m_4res_3f", "m_4res_2f_cnz"):
                f2 = 0.1
                params.add('f2', f2, min=0.01, max=0.4, vary=True)
                
            if self.config["model"] in ("m_4res_3f",):
                f3 = 0.1
                params.add('f3', f3, min=0.01, max=0.4, vary=True)
                
            if self.config['model'] in ("m_4res_f_cnz", "m_4res_2f", "m_4res_3f", "m_4res_2f_cnz"):
                tau_1_hb = 0.1
                params.add('tau_1_hb', tau_1_hb, min=tau_1_hb * 1e-1, max=tau_1_hb * 5e+2, vary=True)
            
    def set_param_values(self, params_config):
        for param_key, param_val in params_config.items():
            self.params[param_key].set(value=param_val)
        self.data_fitted = self.make_prediction(fitted=False)
    
    def _prepare_fitter(self):
        """Subroutine of init procedure to prepare the fit and emcee
        What is mainly done:

        """
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
            
    def make_prediction(self, fitted=True):
        if fitted:
            params = self.result.params
        else:
            params = self.params
        
        data_fitted = {}
        for exp in self.exps:
            if self.config["model"] in MODELS_3_RES:
                ini_exp = [self.init_cond[f'H{exp}'], 
                           self.init_cond[f'D{exp}'], 
                           self.init_cond['NZ'],]
            if self.config["model"] in MODELS_4_RES:
                ini_exp = [self.init_cond[f'H{exp}'], 
                           self.init_cond[f'D{exp}'], 
                           self.init_cond['NZ'],
                           self.init_cond['NZ'],]
            data_fitted[f'H{exp}'] = self.solution(self.data[f'H{exp}']['time'], ini_exp, params)[:, 0]
            data_fitted[f'D{exp}'] = self.solution(self.data[f'D{exp}']['time'], ini_exp, params)[:, 1]
            data_fitted[f'NZ{exp}'] = self.solution(self.data[f'H{exp}']['time'], ini_exp, params)[:, 2]  
        
        return data_fitted 

    def make_fig(self):
        plt.rc('font', family='Helvetica')
        cm = 1/2.54  # cm in inches
        size = 17.1   # in cm
        ver_scale = 0.8
        fig, ax = plt.subplots(2, 2, figsize=(size*cm, ver_scale*size*cm),sharey=True, constrained_layout=True, 
                               gridspec_kw = {'wspace':0})
            
        for exp in self.exps:
            plot_exp(self.data, self.data_fitted,
                    exp, ax, errors=self.config['with_errors'], config=self.config)
            
        fig.tight_layout()
        return fig
    
    def show_plot(self, fitted=True):
        if not fitted:
            self.fig = self.make_fig()
        plt.show()
        
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
        self.data_fitted = self.make_prediction()
        
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
            self.fig = self.make_fig()
            
        # Save plot if requested    
        if save_plot:
            filepath_pdf = os.path.join(self.finpath, 'fit_plot.pdf')
            filepath_png = os.path.join(self.finpath, 'fit_plot.png')
            
            # check if the plot is already exist
            # if yes, it should be deleted
            if os.path.isfile(filepath_pdf):
                os.remove(filepath_pdf)
            
            if os.path.isfile(filepath_png):
                os.remove(filepath_png)
            
            self.fig.savefig(filepath_pdf, bbox_inches='tight')
            self.fig.savefig(filepath_png, bbox_inches='tight')
        
        # Show plot if requested        
        if show_plot:
            plt.show()
        
        self.is_fit_performed = True
        
        ### --- END OF REPORT AND PLOTTING ---

    def emcee(self,
            steps=100,
            progress=True,
            plot=True,
            save_plot=False,
            report=True,
            save_report=False,
            show_walkers=False):
        
        
        if self.is_fit_performed:
            params = self.result.params
        else:
            try:
                params = Parameters()
                load_path = os.path.join(self.finpath, "params.json")
                params.load(open(load_path, "r"))
            except:
                print("You didn't perform fit yet with the given parameters")
                raise
        
        print('--------------------------------------------')
        print('accounting for statistics')
        res = minimize(self.my_residual, method='emcee', nan_policy='omit', burn=30, steps=steps, thin=20,
                        params=params, args=(self.data,), progress=progress)
        
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
            
            # fit_vals = [
            #     res.params.valuesdict()['tau_1'],
            #     res.params.valuesdict()['tau_2'],
            #     res.params.valuesdict()['tau_nz'],
            #     res.params.valuesdict()['c_nz']
            # ]

            fig = corner.corner(res.flatchain, 
                        labels=res.var_names,
                        show_titles=False,
                        # truths=fit_vals,
                        use_math_text=True)
            
            fig.set_tight_layout(True)
            
        if plot:
            self.show_plot()
            
        if save_plot:    
            filepath_pdf = os.path.join(self.finpath, 'emcee_plot.pdf')
            filepath_png = os.path.join(self.finpath, 'emcee_plot.png')
            
            if os.path.isfile(filepath_pdf):
                os.remove(filepath_pdf)
            
            if os.path.isfile(filepath_png):
                os.remove(filepath_png)
            
            fig.savefig(filepath_pdf, bbox_inches='tight')
            fig.savefig(filepath_png, bbox_inches='tight')
            
def calc_everything(model_name="m_3res", given_config=None, fit=True, emcee=False):
    h2o_tempol_list = [
        (2.5, 60),
        (10, 50),
        (10, 60),
        (10, 70),
        (10, 80),
        (17.5, 60),
        (25, 60)
    ]
    dt_list = [True, False]
    
    if given_config:
        fitter = FitterHD(config)
        
        if fit:
            fitter.make_fit(show_plot=False,
                save_plot=True,
                print_report=True,
                save_report=True,
                sample_rate=1)
        
        if emcee:
            fitter.emcee(steps=1000,
                progress=True,
                plot=False,
                save_plot=True,
                report=False,
                save_report=True)
        
        return
        
    config = {
        'h2o_add': 10,     # ul per 100 ul sample
        'conc_tempol': 60,  # mM
        
        # Set is_dt True if TEMPOL is deuterated
        'is_dt': False,
        
        # Set if only H11 and H01 data should be fitted
        # (as if one wouldn't have D data)
        'only_H': False,
        
        # Set True to account for the errors
        'with_errors': True,
        
        # Set the model method
        # All model names start with m_
        "model": model_name,

        # SET FIT METHOD
        # 'leastsq' for a local fit
        # 'differential_evolution' for a global fit
        'fit_method': 'differential_evolution'
    }
    
    for h2o, tempol in h2o_tempol_list:
        for dt in dt_list:
            config['h2o_add'] = h2o
            config['conc_tempol'] = tempol
            config['is_dt'] = dt
            
            fitter = FitterHD(config)
    
            if fit:
                fitter.make_fit(show_plot=False,
                    save_plot=True,
                    print_report=True,
                    save_report=True,
                    sample_rate=1)
            
            if emcee:
                fitter.emcee(steps=300,
                    progress=True,
                    plot=False,
                    save_plot=True,
                    report=False,
                    save_report=True)
                        

if __name__ == '__main__':
    
    # calc_everything(model_name = "m_4res_2f_cnz", fit=False, emcee=True)
    
    config = {
        'h2o_add': 25,     # ul per 100 ul sample
        'conc_tempol': 60,  # mM
        
        # Set is_dt True if TEMPOL is deuterated
        'is_dt': False,
        
        # Set if only H11 and H01 data should be fitted
        # (as if one wouldn't have D data)
        'only_H': False,
        
        # Set True to account for the errors
        'with_errors': True,
        
        # Set the model method
        # All model names start with m_
        "model": "m_4res_2f_cnz",

        # SET FIT METHOD
        # 'leastsq' for a local fit
        # 'differential_evolution' for a global fit
        'fit_method': 'differential_evolution'
    }
    
    fitter = FitterHD(config)
    
    # fitter.set_param_values(
    #     {
    #         "tau_1": 6.7, 
    #         "tau_2": 253,
    #         "tau_nz": 5.01,
    #         "c_nz": 9.47E38,
    #     }
    # )
    
    # fitter.make_fit(show_plot=False,
    #             save_plot=True,
    #             print_report=True,
    #             save_report=True,
    #             sample_rate=1)

    fitter.emcee(steps=300,
                progress=True,
                plot=False,
                save_plot=True,
                report=False,
                save_report=True)
    