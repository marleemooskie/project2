"""
This script includes functions to  adjust the PRISM and ERA data based on the 
AmeriFlux data for each site.
"""
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# This is an example of loading in the cleaned historical data for application
# to adjustment

# Load in daily data
AMF = pd.read_csv("/Users/marleeyork/Documents/project2/data/cleaned/AMF_DD.csv")
AMF.date = pd.to_datetime(AMF.date)

# Calculate daily minimum temperatures
AMF_min = AMF.groupby("Site").apply(lambda g: find_min_temperatures(g.date, g.TA_F)).reset_index()
AMF_min = AMF_min[['Site','date','min_temperature']]

# Calculate daily maximum temperatures
AMF_max = AMF.groupby("Site").apply(lambda g: find_max_temperatures(g.date, g.TA_F)).reset_index()
AMF_max = AMF_max[['Site','date','max_temperature']]

# Load in the historical data for mean, max, and min temperatures
max = pd.read_csv("/Users/marleeyork/Documents/project2/data/cleaned/historical_data_max.csv")
min = pd.read_csv("/Users/marleeyork/Documents/project2/data/cleaned/historical_data_min.csv")
avg = pd.read_csv("/Users/marleeyork/Documents/project2/data/cleaned/historical_data_mean.csv")
AMF = AMF.iloc[:,1:]
AMF = AMF[['Site','date','TA_F']]
max = max.iloc[:,1:]
min = min.iloc[:,1:]
avg = avg.iloc[:,1:]
max.columns = ['Site','date','hist_max','Source_max']
min.columns = ['Site','date','hist_min','Source_min']
avg.columns = ['Site','date','hist_mean','Source_mean']
max.date = pd.to_datetime(max.date)
min.date = pd.to_datetime(min.date)
avg.date = pd.to_datetime(avg.date)

# Merge all these dataframes!
df = pd.merge(max, min, on=['Site','date'], how='inner')
df = pd.merge(df, avg, on=['Site','date'], how='inner')
df = pd.merge(AMF, df, on=['Site','date'], how='left')
df = pd.merge(df, AMF_min, on=['Site','date'], how='inner')
df = pd.merge(df, AMF_max, on=['Site','date'], how='inner')
df = df.dropna()

def fit_sklearn(group: pd.DataFrame, x, y):
    """
    Parameters
    ----------
    group : pd.DataFrame
        DESCRIPTION. The grouped by dataframe that we are going to fit a regression to
        
    x : str
        DESCRIPTION. String column name for AMF data
    
    y : str
        DESCRIPTION. String column name for historical data

    Returns
    -------
    TYPE
        DESCRIPTION. Regression output

    """
    g = group[[x,y]].dropna()
    y = g[y].to_numpy()
    X = g[[x]].to_numpy()
    reg = LinearRegression().fit(X, y)
    
    r2 = reg.score(X, y)
    out = {
        "n": len(y),
        "r2": float(r2),
        "intercept": float(reg.intercept_),
        "coef_x": float(reg.coef_[0])
        }
    
    return pd.Series(out)

def find_historical_bias(final_data):
    """
    Parameters
    ----------
    final_data : TYPE
        DESCRIPTION. This is the dataframe including historical and AMF data for 
        each site. 

    Returns
    -------
    site_historical_fit : TYPE
        DESCRIPTION. This is a dictionary with each site as a key and information
        about regression fit for historical data (PRISM or ERA) to AmeriFlux data

    """
    # Fit mean regression
    mean_results = (
        final_data.groupby("Site", group_keys=False)
        .apply(lambda g: fit_sklearn(g, x="TA_F", y="hist_mean"))
        .reset_index()
        )
    mean_results.columns = ['Site','n_mean','r2_mean','intercept_mean','coef_x_mean']
    
    # Fit the min regression
    min_results = (
        final_data.groupby("Site", group_keys=False)
        .apply(lambda g: fit_sklearn(g, x="min_temperature", y="hist_min"))
        .reset_index()
        )
    min_results.columns = ['Site','n_min','r2_min','intercept_min','coef_x_min']
    
    # Fit the max regression
    max_results = (
        final_data.groupby("Site", group_keys=False)
        .apply(lambda g: fit_sklearn(g, x="max_temperature", y="hist_max"))
        .reset_index()
        )
    max_results.columns = ['Site','n_max','r2_max','intercept_max','coef_x_max']
    
    # Merge all the regression results together
    results = pd.merge(mean_results, min_results, on='Site', how='left')
    results = pd.merge(results, max_results, on='Site', how='left')
    
    return results

def can_we_correct(results: pd.DataFrame, n, r2):
    """
    Description: We only want to adjust sites with 5 years of data and an R^2 > .9.
    This function will tell us which data can be adjusted for each of these sites.
    
    Parameters
    ----------
    results : pd.DataFrame
        DESCRIPTION. DF of regression outputs for PRISM/ERA vs AMF for min, max, and mean temp
        
    n : int
        DESCRIPTION. The minimum days worth of data a site needs to be adjusted
    
    r2 : float
        DESCRIPTION. The minimum R^2 a site regression needs to have to adjust data

    Returns
    -------
    validity_df : pd.DataFrame
        DESCRIPTION. Includes each site and whether we can use the min, max, and mean data

    """
    # Initialize dataframe
    validity_df = pd.DataFrame()
    validity_df['Site'] = results.Site
    
    # Finding those sites with n count and R^2 wanted
    validity_df['Max'] = np.where((results['n_max'] > n) & (results['r2_max'] > r2),1,0)
    validity_df['Min'] = np.where((results['n_min'] > n) & (results['r2_min'] > r2),1,0)
    validity_df['Mean'] = np.where((results['n_mean'] > n) & (results['r2_mean'] > r2),1,0)
    
    return validity_df

def make_adjustment(final_data, results, validity_df):
    """
    Parameters
    ----------
    results : TYPE
        DESCRIPTION. Dataframe from find_historical_bias of the adjustment regressions
    validity_df : TYPE
        DESCRIPTION. Dataframe from can_we_correct of whether or not we can adjust site data

    Returns
    -------
    adjusted_data : TYPE
        DESCRIPTION. Dataframe of sites and adjusted data

    """
    # Initialize dataframe for adjusted data
    adjusted_data = final_data[['Site','date','TA_F','min_temperature','max_temperature']]
    # Loop through each site
    for site in validity_df.Site:
        # Isolate site data
        site_data = final_data[final_data.Site==site]
        # Check if we can adjust the max data
        if (validity_df[validity_df.Site==site].Max.iloc[0] == 1):
            # Find the regression 
            intercept = results[results.Site==site].intercept_max.iloc[0]
            slope = results[results.Site==site].coef_x_max.iloc[0]
            # Use this regression to adjust the data
            adjusted_max = (site_data.hist_max / slope) - intercept
        else:
            # Return NA
            adjusted_max = [np.nan]*len(site_data)
    
    return adjusted_data

fig, ax = plt.subplots()
ax.scatter(site_data.max_temperature, adjusted_max, s=.5)
ax.plot([-30,30],[-30,30],color='red')
plt.show()
