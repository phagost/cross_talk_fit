from fitting_HD import FitterHD

PARAM_NAMES = {
    "3res": ['tau_1', "tau_2", "tau_nz", "c_nz"],
    "3res_f": ['tau_1', "tau_2", "tau_nz", "f1"],
    "3res_2f": ['tau_1', "tau_2", "tau_nz", "f1", "f2"]
}

GLOBAL_CONFIG = {
    "model": "3res",
    "n_fittings": 100,
    "tau_1": 2.32100490,
    "tau_2": 265.832696,
    "tau_nz": 5.52782135,
    "c_nz": 1.1416e+39,
    "f1": 2.043386035145836055e-02,
    "f2": 2.672853905491522908e-01,
}

CONFIG_HD = {
    'h2o_add': 10,      # ul per 100 ul sample
    'conc_tempol': 50,  # mM
    
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

def give_param_dict():
    config = {}
    for param_name in PARAM_NAMES[GLOBAL_CONFIG["model"]]:
        config[param_name] = GLOBAL_CONFIG[param_name]
    return config
    

def main():
    CONFIG_HD['model'] = "m_" + GLOBAL_CONFIG['model'] 
    fitter = FitterHD(CONFIG_HD)
    
    fitter.set_param_values(give_param_dict())
    
    fitter.show_plot(fitted=False)
    
if __name__ == "__main__":
    main()