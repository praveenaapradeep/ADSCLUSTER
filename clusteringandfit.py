# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 18:16:35 2024

@author: Praveena
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def ReadandReturnData(path, filename, indicator):
    """
    Read the filename, return a transposed dataframe df2,
    with Years as column1 and the specified indicator as column2,
    and plot the relationship between them.

    Parameters:
    - path (str): Path to the file.
    - filename (str): Name of the file.
    - indicator (str): Name of the indicator.

    Returns:
    - pd.DataFrame: Transposed dataframe with 'Years' and the specified indicator.
    """
    full_path = f"{path}/{filename}"
    Data = pd.read_csv(full_path, 
