# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 18:16:35 2024

@author: HP
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
    Data = pd.read_csv(full_path, skiprows=4)
    df_Data = pd.DataFrame(Data) 
    df1 = df_Data.loc[df_Data['Country Code'].isin(['GBR'])]
    cols = ['Indicator Name', 'Indicator Code', 'Country Code']
    df_dropped = df1.drop(cols, axis=1)
    df1 = df_dropped.reset_index(drop=True).fillna(0.0)
    df1 = df1.iloc[0:, 0:56]
    df2 = df1.set_index('Country Name').transpose()
    df2['Years'] = df2.index
    temp_cols = df2.columns.tolist()
    new_cols = temp_cols[-1:] + temp_cols[:-1]
    df2 = df2[new_cols]
    df2 = df2.reset_index(drop=True)
    df2 = df2.rename_axis(None, axis=1)
    df2 = df2.rename(columns={'United Kingdom': indicator})

    # Plotting the data
    df2.plot("Years", indicator)
    plt.xlabel("Years")
    plt.ylabel(indicator)
    plt.title(f"{indicator} in United Kingdom")
    plt.show()

    # Returns data frame
    return df2

def exponential(t, n0, g):
    """
    Calculate exponential function with scale factor n0 and growth rate g.

    Parameters:
    - t (float or np.ndarray): Years.
    - n0 (float): Scale factor.
    - g (float): Growth rate.

    Returns:
    - float or np.ndarray: Exponential values.
    """
    t = t - 1960.0
    f = n0 * np.exp(g * t)
    return f

def polynomial_fit(x, *params):
    """
    Calculate polynomial fit.

    Parameters:
    - x (float or np.ndarray): Years.
    - params: Coefficients of the polynomial.

    Returns:
    - float or np.ndarray: Polynomial fit values.
    """
    return np.polyval(params, x)

def err_ranges(x, func, param, sigma):
    """
    Calculate upper and lower limits for the function, parameters, and
    sigmas for a single value or array x.

    Parameters:
    - x (float or np.ndarray): Input values.
    - func (function): Function to calculate limits for.
    - param (tuple): Function parameters.
    - sigma (float or np.ndarray): Sigmas for parameters.

    Returns:
    - tuple: Lower and upper limits.
    """
    import itertools as iter

    lower = func(x, *param)
    upper = lower

    uplow = []  # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper 

# Reading and processing the first dataset
df_clusters = ReadandReturnData('C:\\Users\\HP\\Downloads\\agriland', 'API_AG.LND.AGRI.ZS_DS2_en_csv_v2_6299921.csv','Agricultural land(% of land area)')

# Reading and processing the second dataset for curve fitting
df2 = ReadandReturnData('C:\\Users\\HP\\Downloads\\agrigdp', 'API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_6299253.csv','Agriculture,forestry,andfishing,value added(% of GDP)')

# Normalization
scaler = StandardScaler()

# Normalizing the first dataset
df_clusters[['Years', df_clusters.columns[1]]] = scaler.fit_transform(df_clusters[['Years', df_clusters.columns[1]]])

# Normalizing the second dataset
df2[['Years', df2.columns[1]]] = scaler.fit_transform(df2[['Years', df2.columns[1]]])

# Code for fitting starts here
df2["Years"] = pd.to_numeric(df2["Years"])

# Curve fitting using scipy.optimize.curve_fit for exponential function
param_exp, covar_exp = curve_fit(exponential, df2["Years"], df2[df2.columns[1]], p0=(0, 0))

# Curve fitting using numpy.polyfit for polynomial function
degree = 2  # Choose the degree of the polynomial
param_poly = np.polyfit(df2["Years"], df2[df2.columns[1]], degree)

# Adding column 'Fit_exp' in df2 using the exponential function
df2["Fit_exp"] = exponential(df2["Years"], *param_exp)

# Adding column 'Fit_poly' in df2 using the polynomial function
df2["Fit_poly"] = polynomial_fit(df2["Years"], *param_poly)

# Plotting the graph after fitting for exponential and polynomial fits
plt.figure(figsize=(10, 6))
plt.plot(df2["Years"] + 1960, df2[df2.columns[1]], label="Agriculture, forestry, and fishing, value added (% of GDP)")
plt.plot(df2["Years"] + 1960, df2["Fit_exp"], label="Exponential Fit", linestyle='--')
plt.plot(df2["Years"] + 1960, df2["Fit_poly"], label=f"Polynomial Fit (Degree {degree})", linestyle='--')
plt.xlabel("Year")
plt.ylabel(df2.columns[1])
plt.title(f"Exponential and Polynomial Fits for {df2.columns[1]} in United Kingdom")
plt.legend()
plt.show()

# Passing parameters to err_ranges() function to find the lower and upper range for exponential fit
sigma_exp = np.sqrt(np.diag(covar_exp))
forecast_exp = exponential(df2["Years"], *param_exp)
low_exp, up_exp = err_ranges(df2["Years"], exponential, param_exp, sigma_exp)

# Plotting the confidence range after finding the lower and upper range from err_ranges() for exponential fit
plt.figure(figsize=(10, 6))
plt.plot(df2["Years"] + 1960, df2[df2.columns[1]], label="Agriculture, forestry, and fishing, value added (% of GDP)")
plt.plot(df2["Years"] + 1960, df2["Fit_exp"], label="Exponential Fit", linestyle='--')
plt.fill_between(df2["Years"] + 1960, low_exp, up_exp, color="blue", alpha=0.7, label='Exponential Confidence Range')
plt.xlabel("Year")
plt.ylabel(df2.columns[1])
plt.title(f"Exponential Fit with Confidence Range for {df2.columns[1]} in United Kingdom")
plt.legend()
plt.show()

# Passing parameters to err_ranges() function to find the lower and upper range for polynomial fit
forecast_poly = polynomial_fit(df2["Years"], *param_poly)
low_poly, up_poly = err_ranges(df2["Years"], polynomial_fit, param_poly, sigma_exp)

# Plotting the confidence range after finding the lower and upper range from err_ranges() for polynomial fit
plt.figure(figsize=(10, 6))
plt.plot(df2["Years"] + 1960, df2[df2.columns[1]], label="Agriculture, forestry, and fishing, value added (% of GDP)")
plt.plot(df2["Years"] + 1960, df2["Fit_poly"], label=f"Polynomial Fit (Degree {degree})", linestyle='--')
plt.fill_between(df2["Years"] + 1960, low_poly, up_poly, color="green", alpha=0.7, label=f'Polynomial Confidence Range (Degree {degree})')
plt.xlabel("Year")
plt.ylabel(df2.columns[1])
plt.title(f"Polynomial Fit with Confidence Range for {df2.columns[1]} in United Kingdom")
plt.legend()
plt.show()

# Code for fitting ends here

# Code for Clustering starts here
df_clusters = pd.DataFrame(df_clusters)

# Set up the clusterer using KMeans for clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(df_clusters[['Years']])

# Getting cluster labels and cluster centers
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Scatter plot for the original data points
plt.figure(figsize=(8, 6))
plt.scatter(df_clusters['Years'] + 2000, df_clusters[df_clusters.columns[1]], c=cluster_labels, cmap='viridis', s=50, label='Data Points')

# Scatter plot for cluster centers
plt.scatter(cluster_centers[:, -1], [np.mean(df_clusters[df_clusters.columns[1]][cluster_labels == i]) for i in range(len(cluster_centers))], s=50, linewidths=3, color='red', label='Cluster Centers')

plt.xlabel('Year')
plt.ylabel(f'Normalized {df_clusters.columns[-1]}')
plt.title(f'Clustering of Normalized {df_clusters.columns[-1]} Data with KMeans')
plt.legend()
plt.show()
