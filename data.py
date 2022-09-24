"""This program is made specifically to read the data.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

def read_data(expname='HD_10-60', sample_rate=1, errors=True) -> dict:
    """Read the experimental data.

    Args:
        expname (str, optional): experiment type. Defaults to 'HP_10-60'.
            all the experiments are located in the folder data.
    Returns:
        dict: the dictionary with the data.
            all the data is paired as [time, data].
    """
    start_dir = os.getcwd()
    finpath = os.path.join(start_dir, 'data', expname)
    
    data = {}
    
    files = os.listdir(finpath)
    for f in files:
        exptype = Path(f).stem 
        filename = os.path.join(finpath, f)
        
        df = pd.read_csv(filename)
        time = df['time'].iloc[1:].values.astype(float)
        invT = df['invT'].iloc[1:].values.astype(float)
        if errors:
            err  = df['err'].iloc[1:].values.astype(float)
        else:
            err = 1
        
        data[exptype] = {
            'time': time,
            'data': invT,
            'err': err
        }
        
    # Resample H if needed
    if sample_rate != 1:
        sample_H(data, sample_rate=sample_rate, errors=errors)
            
    return data

def sample_H(data, sample_rate=1, errors=True):
    """_summary_
    The experiments starting with 1 just sample according to the rate
    In the experiments starting with 0 the first 16 points are skipped cause they were recorded
        in log scale and they represent important repolarization statistics 

    Args:
        data (_type_): _description_
    """
    exps = ('11', '10', '01', '00')
    for exp in exps:
        if exp in ('11', '10'):
            data[f'H{exp}']['time'] = data[f'H{exp}']['time'][::sample_rate]
            data[f'H{exp}']['data'] = data[f'H{exp}']['data'][::sample_rate]
            if errors:
                data[f'H{exp}']['err'] = data[f'H{exp}']['err'][::sample_rate]
        if exp in ('01', '00'):
            data[f'H{exp}']['time'] = np.concatenate(
                (data[f'H{exp}']['time'][:16],
                data[f'H{exp}']['time'][16::sample_rate]),
                axis=None
            )
            data[f'H{exp}']['data'] = np.concatenate(
                (data[f'H{exp}']['data'][:16],
                data[f'H{exp}']['data'][16::sample_rate]),
                axis=None
            )
            if errors:
                data[f'H{exp}']['err'] = np.concatenate(
                    (data[f'H{exp}']['err'][:16],
                    data[f'H{exp}']['err'][16::sample_rate]),
                    axis=None
                )           

if __name__ == '__main__':
    exps = ('11', '10', '01', '00')
    
    data = read_data(sample_rate=1)
    
    points_H = 0
    points_D = 0
    for exp in exps:
        points_H += len(data[f'H{exp}']['data'])
        points_D += len(data[f'D{exp}']['data'])
        
    print(f'The number of H points are {points_H}\n'
          f'The number of D points are {points_D}')