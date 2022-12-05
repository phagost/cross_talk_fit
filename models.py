"""
This module provide the models to fit cross-talk experimental data.
"""

import numpy as np

# def model(b, t, params, modelname='2reservroir'):
#     MODEL_DICT = {
#         '2reservoir' : model_two_res 
#     }
    
#     if modelname in MODEL_DICT:
#         return MODEL_DICT[modelname]
#     else:
        

def m_3res(b, t, params):
    """
    Defines the differential equations for the 2 coupled 
    zeeman reservoirs

    Arguments:
        b :  vector of the state variables:
                  b = [b_1, b_2, b_nz]
        t :  time
        params :  vector of the parameters:
                  params = [tau_1, tau_2, tau_nz,
                            c_1, c_2, c_nz]
    """
    
    # Set initial conditions
    b_1, b_2, b_nz = b
    
    # Set initial values
    k_1 = 1 / params['tau_1']
    k_2 = 1 / params['tau_2']
    k_nz = 1 / params['tau_nz']
    c_1 = params['c_1']
    c_2 = params['c_2']
    c_nz = params['c_nz']
    
    # Setting the fraction of hidden spins
    # if f = 0, the model is idential to two spin reservoir model
    
    # The invese temperature of the reservoir
    b_lab = 1 / 4.09 
    
    # Create db_dt = (b_1', b_2', b_nz'):
    db_dt = np.array((
        - k_1 * (b_1-b_nz),      # db_1dt for the first zeeman reservoir
        - k_2 * (b_2-b_nz),      # db_2dt for the second zeeman reservoir
        - (c_1/c_nz) * k_1 * (b_nz - b_1) 
        - (c_2/c_nz) * k_2 * (b_nz - b_2)
        - k_nz * (b_nz-b_lab),  # db_nzdt for non-zeeman reservoir
    ))
    return db_dt

def m_3res_dc(b, t, params):
    """
    Defines the differential equations for the 2 coupled 
    zeeman reservoirs

    Arguments:
        b :  vector of the state variables:
                  b = [b_1, b_2, b_nz]
        t :  time
        params :  vector of the parameters:
                  params = [tau_1, tau_2, tau_nz,
                            c_1, c_2, c_nz]
    """
    
    # Set initial conditions
    b_1, b_2, b_nz = b
    
    # Set initial values
    k_1 = 1 / params['tau_1']
    k_2 = 1 / params['tau_2']
    k_nz = 1 / params['tau_nz']
    c_1 = params['c_1']
    c_2 = params['c_2']
    c_nz = params['c_nz']
    dc = params['dc']
    
    c_1 = c_1 - dc
    c_nz = c_nz + dc
    
    # Setting the fraction of hidden spins
    # if f = 0, the model is idential to two spin reservoir model
    
    # The invese temperature of the reservoir
    b_lab = 1 / 4.09 
    
    # Create db_dt = (b_1', b_2', b_nz'):
    db_dt = np.array((
        - k_1 * (b_1-b_nz),      # db_1dt for the first zeeman reservoir
        - k_2 * (b_2-b_nz),      # db_2dt for the second zeeman reservoir
        - (c_1/c_nz) * k_1 * (b_nz - b_1) 
        - (c_2/c_nz) * k_2 * (b_nz - b_2)
        - k_nz * (b_nz-b_lab),  # db_nzdt for non-zeeman reservoir
    ))
    return db_dt


def m_3res_f(b, t, params):
    """
    Defines the differential equations for the 2 coupled 
    zeeman reservoirs

    Arguments:
        b :  vector of the state variables:
                  b = [b_1, b_2, b_nz]
        t :  time
        params :  vector of the parameters:
                  params = [tau_1, tau_2, tau_nz,
                            c_1, c_2, c_nz]
    """
    
    # Set initial conditions
    b_1, b_2, b_nz = b
    
    # Set initial values
    k_1 = 1 / params['tau_1']
    k_2 = 1 / params['tau_2']
    k_nz = 1 / params['tau_nz']
    c_1 = params['c_1']
    c_2 = params['c_2']
    f = params['f1']
    
    c_1 = c_1 * (1 - f)
    c_nz = f * c_1
    
    # Setting the fraction of hidden spins
    # if f = 0, the model is idential to two spin reservoir model
    
    # The invese temperature of the reservoir
    b_lab = 1 / 4.09 
    
    # Create db_dt = (b_1', b_2', b_nz'):
    db_dt = np.array((
        - k_1 * (b_1-b_nz),      # db_1dt for the first zeeman reservoir
        - k_2 * (b_2-b_nz),      # db_2dt for the second zeeman reservoir
        - (c_1/c_nz) * k_1 * (b_nz - b_1) 
        - (c_2/c_nz) * k_2 * (b_nz - b_2)
        - k_nz * (b_nz-b_lab),  # db_nzdt for non-zeeman reservoir
    ))
    return db_dt

def m_3res_2f(b, t, params):
    """
    Defines the differential equations for the 2 coupled 
    zeeman reservoirs

    Arguments:
        b :  vector of the state variables:
                  b = [b_1, b_2, b_nz]
        t :  time
        params :  vector of the parameters:
                  params = [tau_1, tau_2, tau_nz,
                            c_1, c_2, c_nz]
    """
    
    # Set initial conditions
    b_1, b_2, b_nz = b
    
    # Set initial values
    k_1 = 1 / params['tau_1']
    k_2 = 1 / params['tau_2']
    k_nz = 1 / params['tau_nz']
    c_1 = params['c_1']
    c_2 = params['c_2']
    # c_nz = params['c_nz']
    f1 = params['f1']
    f2 = params['f2']
    
    c_1 = c_1 * (1 - f1)
    c_2 = c_2 * (1 - f2)
    c_nz = f1 * c_1 + f2 * c_2
    
    # Setting the fraction of hidden spins
    # if f = 0, the model is idential to two spin reservoir model
    
    # The invese temperature of the reservoir
    b_lab = 1 / 4.09 
    
    # Create db_dt = (b_1', b_2', b_nz'):
    db_dt = np.array((
        - k_1 * (b_1-b_nz),      # db_1dt for the first zeeman reservoir
        - k_2 * (b_2-b_nz),      # db_2dt for the second zeeman reservoir
        - (c_1/c_nz) * k_1 * (b_nz - b_1) 
        - (c_2/c_nz) * k_2 * (b_nz - b_2)
        - k_nz * (b_nz-b_lab),  # db_nzdt for non-zeeman reservoir
    ))
    return db_dt

def m_3res_f_cnz(b, t, params):
    """
    Defines the differential equations for the 2 coupled 
    zeeman reservoirs

    Arguments:
        b :  vector of the state variables:
                  b = [b_1, b_2, b_nz]
        t :  time
        params :  vector of the parameters:
                  params = [tau_1, tau_2, tau_nz,
                            c_1, c_2, c_nz]
    """
    
    # Set initial conditions
    b_1, b_2, b_nz = b
    
    # Set initial values
    k_1 = 1 / params['tau_1']
    k_2 = 1 / params['tau_2']
    k_nz = 1 / params['tau_nz']
    c_1 = params['c_1']
    c_2 = params['c_2']
    c_nz = params['c_nz']
    f = params['f1']
    
    c_1 = c_1 * (1 - f)
    c_nz = c_nz + f * c_1
    
    # Setting the fraction of hidden spins
    # if f = 0, the model is idential to two spin reservoir model
    
    # The invese temperature of the reservoir
    b_lab = 1 / 4.09 
    
    # Create db_dt = (b_1', b_2', b_nz'):
    db_dt = np.array((
        - k_1 * (b_1-b_nz),      # db_1dt for the first zeeman reservoir
        - k_2 * (b_2-b_nz),      # db_2dt for the second zeeman reservoir
        - (c_1/c_nz) * k_1 * (b_nz - b_1) 
        - (c_2/c_nz) * k_2 * (b_nz - b_2)
        - k_nz * (b_nz-b_lab),  # db_nzdt for non-zeeman reservoir
    ))
    return db_dt


# def m_4res_f_cnz(b, t, params):
#     """
#     Defines the differential equations for the 2 coupled 
#     zeeman reservoirs

#     Arguments:
#         b :  vector of the state variables:
#                   b = [b_1, b_2, b_nz]
#         t :  time
#         params :  vector of the parameters:
#                   params = [tau_1, tau_2, tau_nz,
#                             c_1, c_2, c_nz]
#     """
    
#     # Set initial conditions
#     b_1, b_2, b_nz, b_1_hb = b
    
#     # Set initial values
#     k_1 = 1 / params['tau_1']
#     k_1_hb = 1 / params['tau_1_hb']
#     k_2 = 1 / params['tau_2']
#     k_nz = 1 / params['tau_nz']
#     c_1 = params['c_1']
#     c_2 = params['c_2']
#     c_nz = params['c_nz']
#     f = params['f1']
    
#     c_1_hb = c_1 * f
#     c_1 = c_1 * (1 - f)
    
#     # Setting the fraction of hidden spins
#     # if f = 0, the model is idential to two spin reservoir model
    
#     # The invese temperature of the reservoir
#     b_lab = 1 / 4.09 
    
#     # Create db_dt = (b_1', b_2', b_nz'):
#     db_dt = np.array((
#         - k_1_hb * (b_1-b_1_hb),      # db_1dt for the first zeeman reservoir
#         - k_2 * (b_2-b_nz),      # db_2dt for the second zeeman reservoir
#         - (c_1_hb/c_nz) * k_1 * (b_nz - b_1_hb) 
#         - (c_2/c_nz) * k_2 * (b_nz - b_2)
#         - k_nz * (b_nz-b_lab),  # db_nzdt for non-zeeman reservoir
#         - k_1_hb *(c_1 / c_1_hb) * (b_1_hb-b_1) - k_1 * (b_1_hb-b_nz),  # db_1dt H bulk hidden
#     ))
#     return db_dt


def m_4res_2f(b, t, params):
    """
    Defines the differential equations for the 2 coupled 
    zeeman reservoirs

    Arguments:
        b :  vector of the state variables:
                  b = [b_1, b_2, b_nz]
        t :  time
        params :  vector of the parameters:
                  params = [tau_1, tau_2, tau_nz,
                            c_1, c_2, c_nz]
    """
    
    # Set initial conditions
    b_1, b_2, b_nz, b_1_hb = b
    
    # Set initial values
    k_1 = 1 / params['tau_1']
    k_1_hb = 1 / params['tau_1_hb']
    k_2 = 1 / params['tau_2']
    k_nz = 1 / params['tau_nz']
    c_1 = params['c_1']
    c_2 = params['c_2']
    f1 = params['f1']
    f2 = params['f2']
    
    c_1_hb = c_1 * f1
    c_nz = c_1 * f2
    c_1 = c_1 * (1 - f1 - f2)
    
    # Setting the fraction of hidden spins
    # if f = 0, the model is idential to two spin reservoir model
    
    # The invese temperature of the reservoir
    b_lab = 1 / 4.09 
    
    # Create db_dt = (b_1', b_2', b_nz'):
    db_dt = np.array((
        - k_1_hb * (b_1-b_1_hb),      # db_1dt for the first zeeman reservoir
        - k_2 * (b_2-b_nz),      # db_2dt for the second zeeman reservoir
        - (c_1_hb/c_nz) * k_1 * (b_nz - b_1_hb) 
        - (c_2/c_nz) * k_2 * (b_nz - b_2)
        - k_nz * (b_nz-b_lab),  # db_nzdt for non-zeeman reservoir
        - k_1_hb *(c_1 / c_1_hb) * (b_1_hb-b_1) - k_1 * (b_1_hb-b_nz),  # db_1dt H bulk hidden
    ))
    return db_dt

def m_4res_2f_cnz(b, t, params):
    """
    Defines the differential equations for the 2 coupled 
    zeeman reservoirs

    Arguments:
        b :  vector of the state variables:
                  b = [b_1, b_2, b_nz]
        t :  time
        params :  vector of the parameters:
                  params = [tau_1, tau_2, tau_nz,
                            c_1, c_2, c_nz]
    """
    
    # Set initial conditions
    b_1, b_2, b_nz, b_1_hb = b
    
    # Set initial values
    k_1 = 1 / params['tau_1']
    k_1_hb = 1 / params['tau_1_hb']
    k_2 = 1 / params['tau_2']
    k_nz = 1 / params['tau_nz']
    c_1 = params['c_1']
    c_2 = params['c_2']
    c_nz = params['c_nz']
    f1 = params['f1']
    f2 = params['f2']
    
    c_1_hb = c_1 * f1
    c_nz = c_nz + c_1 * f2
    c_1 = c_1 * (1 - f1 - f2)
    
    # Setting the fraction of hidden spins
    # if f = 0, the model is idential to two spin reservoir model
    
    # The invese temperature of the reservoir
    b_lab = 1 / 4.09 
    
    # Create db_dt = (b_1', b_2', b_nz'):
    db_dt = np.array((
        - k_1_hb * (b_1-b_1_hb),      # db_1dt for the first zeeman reservoir
        - k_2 * (b_2-b_nz),      # db_2dt for the second zeeman reservoir
        - (c_1_hb/c_nz) * k_1 * (b_nz - b_1_hb) 
        - (c_2/c_nz) * k_2 * (b_nz - b_2)
        - k_nz * (b_nz-b_lab),  # db_nzdt for non-zeeman reservoir
        - k_1_hb *(c_1 / c_1_hb) * (b_1_hb-b_1) - k_1 * (b_1_hb-b_nz),  # db_1dt H bulk hidden
    ))
    return db_dt

def m_4res_3f(b, t, params):
    """
    Defines the differential equations for the 2 coupled 
    zeeman reservoirs

    Arguments:
        b :  vector of the state variables:
                  b = [b_1, b_2, b_nz]
        t :  time
        params :  vector of the parameters:
                  params = [tau_1, tau_2, tau_nz,
                            c_1, c_2, c_nz]
    """
    
    # Set initial conditions
    b_1, b_2, b_nz, b_1_hb = b
    
    # Set initial values
    k_1 = 1 / params['tau_1']
    k_1_hb = 1 / params['tau_1_hb']
    k_2 = 1 / params['tau_2']
    k_nz = 1 / params['tau_nz']
    c_1 = params['c_1']
    c_2 = params['c_2']
    f1 = params['f1']
    f2 = params['f2']
    f3 = params['f3']
    
    c_1_hb = c_1 * f1
    c_nz = c_1 * f2 + c_2 * f3
    c_1 = c_1 * (1 - f1 - f2)
    c_2 = c_2 * (1 - f3)
    
    # Setting the fraction of hidden spins
    # if f = 0, the model is idential to two spin reservoir model
    
    # The invese temperature of the reservoir
    b_lab = 1 / 4.09 
    
    # Create db_dt = (b_1', b_2', b_nz'):
    db_dt = np.array((
        - k_1_hb * (b_1-b_1_hb),      # db_1dt for the first zeeman reservoir
        - k_2 * (b_2-b_nz),      # db_2dt for the second zeeman reservoir
        - (c_1_hb/c_nz) * k_1 * (b_nz - b_1_hb) 
        - (c_2/c_nz) * k_2 * (b_nz - b_2)
        - k_nz * (b_nz-b_lab),  # db_nzdt for non-zeeman reservoir
        - k_1_hb *(c_1 / c_1_hb) * (b_1_hb-b_1) - k_1 * (b_1_hb-b_nz),  # db_1dt H bulk hidden
    ))
    return db_dt



def ml_3res_2hid_nocnz(b, t, params):
    """
    Defines the differential equations for the 2 coupled 
    zeeman reservoirs

    Arguments:
        b :  vector of the state variables:
                  b = [b_1, b_2, b_nz]
        t :  time
        params :  vector of the parameters:
                  params = [tau_1, tau_2, tau_nz,
                            c_1, c_2, c_nz]
    """
    
    # Set initial conditions
    b_1, b_2, b_nz = b
    
    # Set initial values
    k_1 = 1 / params['tau_1']
    k_2 = 1 / params['tau_2']
    k_nz = 1 / params['tau_nz']
    c_1 = params['c_1']
    c_2 = params['c_2']
    c_nz = params['c_nz']
    
    f1 = params['f1']
    f2 = params['f2']
    
    # Setting the fraction of hidden spins
    # if f = 0, the model is idential to two spin reservoir model
    
    # The invese temperature of the reservoir
    b_lab = 1 / 4.09 
    
    # corrected heat capcasity ratios
    kappa_1 = (1-f1)/(f1+f2*c_2/c_1)
    kappa_2 = (1-f2)/(f2+f1*c_1/c_2)
    
    k_nz = k_nz * c_nz / (f1*c_1 + f2*c_2)
    
    # Create db_dt = (b_1', b_2', b_nz'):
    db_dt = np.array((
        - k_1 * (b_1-b_nz),      # db_1dt for the first zeeman reservoir
        - k_2 * (b_2-b_nz),      # db_2dt for the second zeeman reservoir
        - kappa_1 * k_1 * (b_nz - b_1) 
        - kappa_2 * k_2 * (b_nz - b_2)
        - k_nz * (b_nz-b_lab),  # db_nzdt for non-zeeman reservoir
    ))
    return db_dt

def m_3res_d(b, t, params):
    """
    Defines the differential equations for the 2 coupled 
    zeeman reservoirs

    Arguments:
        b :  vector of the state variables:
                  b = [b_1, b_2, b_nz]
        t :  time
        params :  vector of the parameters:
                  params = [tau_1, tau_2, tau_nz,
                            c_1, c_2, c_nz]
    """
    
    # Set initial conditions
    b_1, b_2, b_nz = b
    
    # Set initial values
    k_1 = 1 / params['tau_1']
    k_2 = 1 / params['tau_2']
    k_nz = 1 / params['tau_nz']
    c_1 = params['c_1']
    c_2 = params['c_2']
    c_nz = params['c_nz']
    
    # Setting the fraction of hidden spins
    # if f = 0, the model is idential to two spin reservoir model
    
    # The invese temperature of the reservoir
    b_lab = 1 / 4 
    
    # Create db_dt = (b_1', b_2', b_nz'):
    db_dt = np.array((
        k_1*(b_nz),      # db_1dt for the first zeeman reservoir
        k_2*(b_nz),      # db_2dt for the second zeeman reservoir
        (c_1 / c_nz)*k_1*(b_1 - b_nz) 
        + (c_2 / c_nz)*k_2*(b_2 - b_nz)
        + k_nz*(b_lab - b_nz),  # db_nzdt for non-zeeman reservoir
    ))
    return db_dt

def m_3res_1hid(b, t, params):
    """
    Defines the differential equations for the 2 coupled 
    zeeman reservoirs

    Arguments:
        b :  vector of the state variables:
                  b = [b_1, b_2, b_nz]
        t :  time
        params :  vector of the parameters:
                  params = [tau_1, tau_2, tau_nz,
                            c_1, c_2, c_nz]
    """
    
    # Set initial conditions
    b_1, b_2, b_nz, b_1h = b
    
    
    # Set initial values
    k_1 = 1 / params['tau_1']
    k_2 = 1 / params['tau_2']
    k_nz = 1 / params['tau_nz']
    c_1 = params['c_1']
    c_2 = params['c_2']
    c_nz = params['c_nz']
    
    # Setting the fraction of hidden spins
    # if f = 0, the model is idential to two spin reservoir model
    # Here we set that relaxation properties of the hidden reservoir is comparable
    # with the DD, but this is not 100% true
    f1 = params['f1']
    k_1h = params['tau_1h']
    
    # The invese temperature of the reservoir
    b_lab = 1 / 4 
    
    # Create db_dt = (b_1', b_2', b_nz'):
    db_dt = np.array((
        - k_1 * (b_1-b_nz),      # db_1dt for the first zeeman reservoir
        - k_2 * (b_2-b_nz),      # db_2dt for the second zeeman reservoir
        - (1-f1) * (c_1/c_nz) * k_1 * (b_nz-b_1) 
        - (c_2/c_nz) * k_2 * (b_nz-b_2)
        - k_nz * (b_nz-b_lab)
        - f1 * (c_1/c_nz) * k_1h*(b_nz - b_1h),  # db_nzdt for non-zeeman reservoir
        - k_1h * (b_1h - b_nz),                  # db_1hdt for hidden zeeman reservoir of the second nuclei
    ))
    return db_dt

def m_3res_2hid(b, t, params):
    """
    Defines the differential equations for the 2 coupled 
    zeeman reservoirs

    Arguments:
        b :  vector of the state variables:
                  b = [b_1, b_2, b_nz]
        t :  time
        params :  vector of the parameters:
                  params = [tau_1, tau_2, tau_nz,
                            c_1, c_2, c_nz]
    """
    
    # Set initial conditions
    b_1, b_2, b_nz, b_1h, b_2h = b
    
    
    # Set initial values
    k_1 = 1 / params['tau_1']
    k_2 = 1 / params['tau_2']
    k_nz = 1 / params['tau_nz']
    c_1 = params['c_1']
    c_2 = params['c_2']
    c_nz = params['c_nz']
    
    # Setting the fraction of hidden spins
    # if f = 0, the model is idential to two spin reservoir model
    # Here we set that relaxation properties of the hidden reservoir is comparable
    # with the DD, but this is not 100% true
    f1 = params['f1']
    k_1h = params['tau_1h']
    f2 = params['f2']
    k_2h = params['tau_2h']
    
    
    # The invese temperature of the reservoir
    b_lab = 1 / 4 
    
    # Create db_dt = (b_1', b_2', b_nz'):
    db_dt = np.array((
        - k_1 * (b_1-b_nz),      # db_1dt for the first zeeman reservoir
        - k_2 * (b_2-b_nz),      # db_2dt for the second zeeman reservoir
        - (1-f1) * (c_1/c_nz) * k_1 * (b_nz-b_1) 
        - (1-f2) * (c_2/c_nz) * k_2 * (b_nz-b_2)
        - k_nz * (b_nz-b_lab)
        - f1 * (c_1/c_nz) * k_1h * (b_nz-b_1h)
        - f2 * (c_1/c_nz) * k_2h * (b_nz-b_2h),  # db_nzdt for non-zeeman reservoir
        - k_1h * (b_1h-b_nz),                    # db_1hdt for hidden zeeman reservoir of the second nuclei
        - k_2h * (b_2h-b_nz),                    # db_1hdt for hidden zeeman reservoir of the second nuclei
    ))
    return db_dt

def m_4res_X(b, t, params):
    """
    Defines the differential equations for the 2 coupled 
    zeeman reservoirs

    Arguments:
        b :  vector of the state variables:
                  b = [b_1, b_2, b_nz]
        t :  time
        params :  vector of the parameters:
                  params = [tau_1, tau_2, tau_nz,
                            c_1, c_2, c_nz]
    """
    
    # Set initial conditions
    b_1, b_2, b_3, b_nz = b
    
    # Set initial values
    k_1 = 1 / params['tau_1']
    k_2 = 1 / params['tau_2']
    k_3 = 1 / params['tau_3']
    k_nz = 1 / params['tau_nz']
    c_1 = params['c_1']
    c_2 = params['c_2']
    c_3 = params['c_3']
    c_nz = params['c_nz']
    
    
    # The invese temperature of the reservoir
    b_lab = 1 / 4 
    
    # Create db_dt = (b_1', b_2', b_nz'):
    db_dt = np.array((
        - k_1 * (b_1-b_nz),      # db_1dt for the first zeeman reservoir
        - k_2 * (b_2-b_nz),      # db_2dt for the second zeeman reservoir
        - k_3 * (b_3-b_nz),      # db_3dt for the third zeeman reservoir
        - (c_1/c_nz) * k_1 * (b_nz-b_1) 
        - (c_2/c_nz) * k_2 * (b_nz-b_2)
        - (c_3/c_nz) * k_3 * (b_nz-b_3)
        - k_nz * (b_nz-b_lab),  # db_nzdt for non-zeeman reservoir
    ))
    return db_dt


def m_3res_f_cnz_X(b, t, params):
    """
    Defines the differential equations for the 2 coupled 
    zeeman reservoirs

    Arguments:
        b :  vector of the state variables:
                  b = [b_1, b_2, b_nz]
        t :  time
        params :  vector of the parameters:
                  params = [tau_1, tau_2, tau_nz,
                            c_1, c_2, c_nz]
    """
    
    # Set initial conditions
    b_1, b_2, b_3, b_nz = b
    
    # Set initial values
    k_1 = 1 / params['tau_1']
    k_2 = 1 / params['tau_2']
    k_3 = 1 / params['tau_3']
    k_nz = 1 / params['tau_nz']
    c_1 = params['c_1']
    c_2 = params['c_2']
    c_3 = params['c_3']
    c_nz = params['c_nz']
    f = params['f1']
    
    c_1 = c_1 * (1 - f)
    c_nz = c_nz + f * c_1
    
    # Setting the fraction of hidden spins
    # if f = 0, the model is idential to two spin reservoir model
    
    # The invese temperature of the reservoir
    b_lab = 1 / 4.09 
    
    # Create db_dt = (b_1', b_2', b_nz'):
    db_dt = np.array((
        - k_1 * (b_1-b_nz),      # db_1dt for the first zeeman reservoir
        - k_2 * (b_2-b_nz),      # db_2dt for the second zeeman reservoir
        - k_3 * (b_3-b_nz),      # db_3dt for the third zeeman reservoir
        - (c_1/c_nz) * k_1 * (b_nz - b_1) 
        - (c_2/c_nz) * k_2 * (b_nz - b_2)
        - (c_3/c_nz) * k_3 * (b_nz - b_3)
        - k_nz * (b_nz-b_lab),  # db_nzdt for non-zeeman reservoir
    ))
    return db_dt

def m_4res_1hid(b, t, params):
    """
    Defines the differential equations for the 2 coupled 
    zeeman reservoirs

    Arguments:
        b :  vector of the state variables:
                  b = [b_1, b_2, b_nz]
        t :  time
        params :  vector of the parameters:
                  params = [tau_1, tau_2, tau_nz,
                            c_1, c_2, c_nz]
    """
    
    # Set initial conditions
    b_1, b_2, b_3, b_nz, b_1h = b
    
    # Set initial values
    k_1 = 1 / params['tau_1']
    k_2 = 1 / params['tau_2']
    k_3 = 1 / params['tau_3']
    k_nz = 1 / params['tau_nz']
    c_1 = params['c_1']
    c_2 = params['c_2']
    c_3 = params['c_3']
    c_nz = params['c_nz']
    
    # Here we set that relaxation properties of the hidden reservoir is comparable
    # with the DD, but this is not 100% true
    f1 = params['f1']
    k_1h = params['tau_1h']
    
    # The invese temperature of the reservoir
    b_lab = 1 / 4 
    
    # Create db_dt = (b_1', b_2', b_nz'):
    db_dt = np.array((
        - k_1 * (b_1-b_nz),      # db_1dt for the first zeeman reservoir
        - k_2 * (b_2-b_nz),      # db_2dt for the second zeeman reservoir
        - k_3 * (b_3-b_nz),      # db_3dt for the third zeeman reservoir
        - (1-f1) * (c_1/c_nz) * k_1 * (b_nz-b_1) 
        - (c_2/c_nz) * k_2 * (b_nz-b_2)
        - (c_3/c_nz) * k_3 * (b_nz-b_3)
        - k_nz * (b_nz-b_lab),  # db_nzdt for non-zeeman reservoir
        - k_1h * (b_1h-b_nz),
    ))
    return db_dt

if __name__ == '__main__':
    """Plotting some solutions with the close to reality for either HP or HD data.
    """
    from scipy.integrate import odeint
    from matplotlib import pyplot as plt
    from capacity import calc_capacity_zeeman_H,\
                         calc_capacity_zeeman_P, \
                         calc_capacity_zeeman_D, \
                         calc_capacity_nz
    
    MODEL = '2RES'
    
    if MODEL == '2RES':
        composition = {
            'h2o': 10,      # ul per 100 ul
            'd2o': 40,     # ul per 100 ul
            'k2hpo4': 0.5,     # M
            'TEMPOL': 60 # mM
        }
        
        params = {
            # Initial condition 01
            'b1_01': 0,
            'b2_01': 15,
            'bss_01': 15,
            
            # Initial condition 10
            'b1_10': 15,
            'b2_10': 0,
            'bss_10': 15,
            
            'tau_1': 30,
            'tau_2': 200,
            'tau_nz': 5,
            'c_1': calc_capacity_zeeman_H(composition['h2o']),
            'c_2': calc_capacity_zeeman_D(composition['d2o']),
            'c_nz': calc_capacity_nz(composition['TEMPOL'])*10e2,
        }
        
        time = np.linspace(0, 500, 2000)
        
        # Plotting 01 results
        ini_01 = [params['b1_01'], params['b2_01'], params['bss_01']]
        sol_01 = odeint(m_3res, ini_01, time, args=(params,))
        
        # Plotting 10 results
        ini_10 = [params['b1_10'], params['b2_10'], params['bss_10']]
        sol_10 = odeint(m_3res, ini_10, time, args=(params,))


        fig, ax = plt.subplots(1, 2)
        
        ax[0].plot(time, sol_01[:, 0], 'r', label = r'$^{1}H$ real')
        ax[0].plot(time, sol_01[:, 1], 'b', label = r'$^{31}P$ real')
        ax[0].plot(time, sol_01[:, 2], 'g--', label = r'DD real')
        ax[0].set_title(r'01', fontsize = 20)
        ax[0].set_xlabel('time / sec', fontsize = 17.5)
        ax[0].set_ylabel(r'$\beta$ / $K^{-1}$', fontsize = 17.5)
        
        ax[1].plot(time, sol_10[:, 0], 'r', label = r'$^{1}H$ real')
        ax[1].plot(time, sol_10[:, 1], 'b', label = r'$^{31}P$ real')
        ax[1].plot(time, sol_10[:, 2], 'g--', label = r'DD real')
        ax[1].set_title(r'10', fontsize = 20)
        ax[1].set_xlabel('time / sec', fontsize = 17.5)
        # ax[1].set_ylabel(r'$\beta$ / $K^{-1}$', fontsize = 17.5)
        
        
        plt.legend(fontsize = 15)
        print(params['c_1'])
        print(params['c_2'])
        plt.show()
        