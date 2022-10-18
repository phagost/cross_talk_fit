from tqdm import tqdm
import numpy as np
import random
import concurrent.futures as ccf
import os
import corner

from fitting_HD import FitterHD
from capacity import calc_capacity_nz

PARAM_NAMES = {
    "3res": ['tau_1', "tau_2", "tau_nz", "c_nz"],
    "3res_f": ['tau_1', "tau_2", "tau_nz", "f1"],
    "3res_f_cnz": ['tau_1', "tau_2", "tau_nz", "f1", "c_nz"],
    "3res_2f": ['tau_1', "tau_2", "tau_nz", "f1", "f2"],
    "4res_f_стя": ['tau_1', "tau_2", "tau_nz", "f", "с_nz", "tau_1_hb"],
}

GLOBAL_CONFIG = {
    "model": "3res_f_cnz",
    "n_fittings": 25,
    "tau_1": {
        "min": 0.1,
        "max": 200,
    },
    "tau_2": {
        "min": 50,
        "max": 600,
    },
    "tau_nz": {
        "min": 5e-3,
        "max": 1e+1,        
    },
    "c_nz": {
        "min": calc_capacity_nz(60),
        "max": calc_capacity_nz(60) * 10e+3,
    },
    "tau_1_hb":{
        "min": 1e-2,
        "max": 10 
    },
    "f1": {
        "min": 0.01,
        "max": 0.99,
    },
    "f2": {
        "min": 0.01,
        "max": 0.99,
    },
}

def get_param_val(fitter):
    res = []
    for param_name in PARAM_NAMES[GLOBAL_CONFIG["model"]]:
        res.append(fitter.result.params[param_name].value)
    res.append(fitter.result.chisqr)
    return res

def gen_params():
    config = {}
    for param_name in PARAM_NAMES[GLOBAL_CONFIG["model"]]:
        config[param_name] = random.uniform(
            GLOBAL_CONFIG[param_name]["min"],
            GLOBAL_CONFIG[param_name]["max"]
        )
    return config

def make_fit(fitter):
    fitter.set_param_values(gen_params())
    fitter.make_fit(show_plot=False,
                    save_plot=False,
                    print_report=False,
                    save_report=False,
                    sample_rate=1)
    return get_param_val(fitter)

def gather_params(n_fittings, config):
    fitter_HD = FitterHD(config)
    
    res = []
    with ccf.ProcessPoolExecutor() as executor:
        results = [ executor.submit(make_fit, fitter_HD) for _ in range(n_fittings)] 
        
        for f in tqdm(ccf.as_completed(results)):
            res.append(f.result())
            
    return res
        
def make_name(config):
    # Strange way to find the amount of additional water
    if not isinstance(config['h2o_add'], int):
        first = str(int(config['h2o_add']))
        second = str(int((config['h2o_add'] * 10) % 10))
        h2o_add_str = first + '&' + second
    else:
        h2o_add_str = str(config['h2o_add'])
    
    
    TEMPOL = config['conc_tempol']
    exp_name = f'HD_{h2o_add_str}-{TEMPOL}'
    
    if config['is_dt']:
        exp_name = exp_name + '_dt'
            
    return exp_name        

def plt_best(best_vals, config):
    param_config = {}
    for name, val in zip(PARAM_NAMES[GLOBAL_CONFIG["model"]], best_vals):
        param_config[name] = val
    
    fitter = FitterHD(config)
    fitter.set_param_values(param_config)
    fitter.show_plot(fitted=False)
    
def save_results(res, config):
    exp_name = make_name(config)
    
    start_dir = os.getcwd()
    dir_name = os.path.join(start_dir, 'models', GLOBAL_CONFIG["model"], exp_name)
    if not os.path.isdir(dir_name):
            # os.mkdir(dir_name)
            os.makedirs(dir_name, exist_ok=True)
    
    np_res = np.array(res)
    
    np_sorted = np_res[np_res[:, -1].argsort()]
    
    means = np_sorted[:, :-1].mean(axis=0)
    stds = np_sorted[:, :-1].std(axis=0)
    
    np.savetxt(os.path.join(dir_name, "means"), means, delimiter=',')
    np.savetxt(os.path.join(dir_name, "stds"), stds, delimiter=',')
    np.savetxt(os.path.join(dir_name, "data"), np_sorted, delimiter=',')
    
    # print(np_sorted[:, :-1].shape)
    # print(len(GLOBAL_CONFIG["param_names"]))

    fig = corner.corner(np_sorted[:, :-1], 
        labels=PARAM_NAMES[GLOBAL_CONFIG["model"]]
        )
    
    fig.set_tight_layout(True)
    fig.savefig(os.path.join(dir_name, "corner"), bbox_inches='tight')
    
    plt_best(np_sorted[0, :-1], config)
    
    
def main():
    config_HD = {
        'h2o_add': 2.5,      # ul per 100 ul sample
        'conc_tempol': 60,  # mM
        
        # Set is_dt True if TEMPOL is deuterated
        'is_dt': True,
        
        # Set if only H11 and H01 data should be fitted
        # (as if one wouldn't have D data)
        'only_H': False,
        
        # Set True to account for the errors
        'with_errors': False,

        # SET FIT METHOD
        # 'leastsq' for a local fit
        # 'differential_evolution' for a global fit
        'fit_method': 'differential_evolution'
    }
    config_HD['model'] = 'm_' + GLOBAL_CONFIG['model']
        
    res = gather_params(GLOBAL_CONFIG["n_fittings"], config_HD)
    save_results(res, config_HD)
    
if __name__ == "__main__":
    main()