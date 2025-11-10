'''
In this script, I will do QAQC including:
    - inspecting site variable values and removing missing areas
    - correcting negative values that shouldn't be there
    - comparing temperature values in PRISM and AmeriFlux data for consistent
    
Currently, I am compring quantiles between PRISM and AMF for max temperatures. 
They align at most the sites, except two, which I am investigating month specific 
misalignment. Next I need to do this for minimum and average temperatures, but 
first I will need to acquire the PRISM for these measrements.

Also, we need PRISM data for Canada too.
'''

import os
os.chdir("/Users/marleeyork/Documents/project2")
from load_data import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
pd.set_option('display.max_columns',300)
pd.set_option('display.max_rows',100)

# Loading in the AmeriFlux data across all sites with soil water content
# This automatically loads selected for columns, defined in loadAMF
df = loadAMF(path='/Users/marleeyork/Documents/project2/data/AMFdataDD',
                 measures=['TIMESTAMP','TA_F','SW_IN_F','VPD_F','P_F','NEE_VUT_REF','RECO_NT_VUT_REF','GPP_NT_VUT_REF','SWC_F_MDS_1'])

df_hourly = loadAMF(path='/Users/marleeyork/Documents/project2/data/AMFdata_HH',
                 measures=['TIMESTAMP_START','TA_F'])

# Load the IGBP data and merge to df
site_data = pd.read_csv("/Users/marleeyork/Documents/project2/data/site_list.csv",encoding='latin1')
IGBP = site_data[['Site ID','Vegetation Abbreviation (IGBP)']]
IGBP.columns = ['Site','IGBP']
df = pd.merge(df,IGBP,on="Site",how="inner").drop_duplicates()

df.columns
df.shape
df.Site.unique()
len(df.Site.unique())
df.IGBP.unique()
###############################################################################
##                                 Edits                                     ##
###############################################################################

# Drop any rows with values that are -9999 (new shape is (190298,11))
mask = df.apply(lambda col: col == -9999).any(axis=1)
df = df[~mask]

# Dropping US-Ne1, not sure why its in here
df = df[df['IGBP']!='CRO']

# If GPP is negative, set it to 0
df.loc[df['GPP_NT_VUT_REF']<0,'GPP_NT_VUT_REF'] = 0

# Now we will do site specific filtering
# Initializing a drop list that we will fill with indices to remove
drop_list = []

# SRG days before 05/02/2018 (drop_list.len == 70)
drop_list.extend(df[(df['Site'] == 'US-SRG') & (df['TIMESTAMP'] < '2008-05-02')].index)

# SRM days after 2017-05-01 (drop_list.len == 2871)
drop_list.extend(df[(df['Site'] == 'US-SRM') & (df['TIMESTAMP'] > '2017-05-01')].index)

# LP1 days after 2016-12-01 (drop_list.len == 3997)
drop_list.extend(df[(df['Site'] == 'CA-LP1') & (df['TIMESTAMP'] > '2016-12-01')].index)

# Me2 days after 2021-10-01 (drop_list.len == 4295)
drop_list.extend(df[(df['Site'] == 'US-Me2') & (df['TIMESTAMP'] > '2021-10-01')].index)

# BZS days prior to 2015-01-01 (drop_list.len == 4582)
drop_list.extend(df[(df['Site'] == 'US-BZS') & (df['TIMESTAMP'] < '2015-01-01')].index)

# Syv days prior to 2012-01-01 (drop_list.len == 6926)
drop_list.extend(df[(df['Site'] == 'US-Syv') & (df['TIMESTAMP'] < '2012-01-01')].index)

# KFS days prior to 2009-01-01 (drop_list.len == 7292)
drop_list.extend(df[(df['Site'] == 'US-KFS') & (df['TIMESTAMP'] < '2009-01-01')].index)

# Ton days after to 2020-05-01 (drop_list.len == 8997)
drop_list.extend(df[(df['Site'] == 'US-Ton') & (df['TIMESTAMP'] > "2020-05-01")].index)


# These are corrections made to cropland (CRO) sites, which have been removed
# Mo1 days before 2016-01-01 (drop_list.len == 2978)
# drop_list.extend(df[(df['Site'] == 'US-Mo1') & (df['TIMESTAMP'] < '2016-01-01')].index)

# Drop the drop list! (df.shape == (161011,10))
df = df.drop(drop_list)

# Eventually I need to write this to a dataframe, but I'm not sure whats going on 
# with some of my sites for now.

###############################################################################
##                                  QAQC                                     ##
###############################################################################
vars_to_plot = [
    "TA_F",
    "SW_IN_F",
    "VPD_F",
    "P_F",
    "SWC_F_MDS_1",
    "GPP_NT_VUT_REF"
]

for site in df.Site.unique():
    flux_data = df[df['Site']==site]
    fig, ax = plt.subplots(3, 2, figsize=(10, 8))
    ax = ax.flatten()

    for a, var in zip(ax, vars_to_plot):
        a.scatter(flux_data.TIMESTAMP, flux_data[var], s=0.5)
        a.set_title(f"{site} {var}", fontsize=7)
        a.set_xlabel("Date", fontsize=6)
        a.tick_params(axis='x', rotation=45, labelsize=6)
        a.tick_params(axis='y', labelsize=6)
        #a.axvline(x=pd.to_datetime("2020-05-01"),color="red")


    plt.tight_layout()
    plt.show()
    
    input("Press [Enter] to continue...")
    
###############################################################################
##                       TEMPERATURE CHECKS                                  ##
###############################################################################

# Comparing max temperature PRISM to AmeriFlux
# Right now, I have this available across all 33 sites!

# Load in the data
historical_max = pd.read_csv("/Users/marleeyork/Documents/project2/data/PRISM/extracted_daily_climate_data_tmax.csv")

# Drop any sites that aren't in df
included_sites = df.Site.unique()
included_sites = np.insert(included_sites,0,'date')
historical_max = historical_max[included_sites]

# Findig which sites have missing values
search_value = -9999
columns_with_value = []
for col in historical_max.columns:
    # Check if the search_value exists in the current column
    if historical_max[col].astype(str).str.contains(str(search_value)).any():
        columns_with_value.append(col)
        
print(f"Sites that we don't have PRISM data include: {columns_with_value}")

# These columns are completely missing all PRISM temperature, interesting
# As of now, its the Canada sites we are missing data for (PRISM doesn't go to Canada)
# and two US sites that are in Alaska
historical_max[columns_with_value]

# Starting off with the ones that we do know
df_max = historical_max.drop(columns=columns_with_value)

# Pivoting df_max longer
df_max = df_max.melt(id_vars=['date'],var_name='Site',value_name='max_PRISM')

# Now I am going to calculate the daily maximum temperature for each of my sites
# using my function as done in my heatwave definition
AMF_max = pd.DataFrame(columns=['Site','date','max_temperature'])
for site in df_max.Site.unique():
    this_site = df_hourly[df_hourly['Site']==site]
    this_site_temp = find_max_temperatures(this_site.TIMESTAMP_START,this_site.TA_F)
    this_site_temp['Site'] = [site] * this_site_temp.shape[0]
    # concatenate with entire dataframe
    AMF_max = pd.concat([AMF_max,this_site_temp])
    
# Convert df_max date to a datetime variable
df_max.date = pd.to_datetime(df_max.date)

# Merging df and df max
# df.shape = (159306, 11)
# df_max.shape = (15705, 24)
# df_max after the merge = (140984, 4)
df_max = pd.merge(AMF_max, df_max, on=['Site','date'])

# Lets look at the overall correlation between the PRISM and AmeriFlux data now
fig, ax = plt.subplots()
plt.scatter(df_max.max_temperature,df_max.max_PRISM, s=.5,alpha=.1)
plt.xlabel("AMF Max Temperature")
plt.ylabel("PRISM Max Temperature")
plt.show()

# Checking the overall correlation coefficient
# Correlation is 93%, which is pretty good
np.corrcoef(df_max.max_temperature,df_max.max_PRISM)

# Checking site by site correlation to look for any anomalies
# So far ONA and KFS have the weakest correlation, which is mid 80s. Everything
# else is fine.
for site in df_max.Site.unique():
    this_site = df_max[df_max['Site']==site]
    print(f"Correlation for site {site} is {np.corrcoef(this_site.max_temperature,this_site.max_PRISM)[0][1]}")

# Finding the 95th percentile for each
for site in df_max.Site.unique():
    this_site = df_max[df_max['Site']==site]
    # Calculate the 95th quantile overall
    AMF_95 = np.quantile(this_site.max_temperature,.95)
    PRISM_95 = np.quantile(this_site.max_PRISM,.95)
    print(f"95th quantiles at site {site} are {AMF_95} and {PRISM_95}.")
    
# Find correlation between daily 95th quantile values
# So far, these 
daily_quantiles = pd.DataFrame(columns=['Site','month_day','AMF_quantiles','PRISM_quantiles'])
for site in df_max.Site.unique():
    this_site = df_max[df_max['Site']==site]
    # Calculate window quantiles for each
    AMF_daily_95 = moving_window_quantile(this_site.date,this_site.max_temperature,.95, 15)
    AMF_daily_95.columns = ["month_day","AMF_quantiles"]
    PRISM_daily_95 = moving_window_quantile(this_site.date,this_site.max_PRISM,.95,15)
    PRISM_daily_95.columns = ["month_day","PRISM_quantiles"]
    # Merge these together
    window_quantiles = pd.merge(AMF_daily_95,PRISM_daily_95,on='month_day',how='inner')
    # Add a site columns
    window_quantiles['Site'] = [site] * window_quantiles.shape[0]
    # Stack it onto the dataframe across all sites
    daily_quantiles = pd.concat([daily_quantiles,window_quantiles])
    # Print off the correlation coefficient
    print(f"Correlation coefficient for site {site} is {np.corrcoef(daily_quantiles.AMF_quantiles,daily_quantiles.PRISM_quantiles)[0][1]}")
    
# Plotting the AMF and PRISM 95th moving quantiles
fig, ax = plt.subplots()
sb.lmplot(x='AMF_quantiles', y='PRISM_quantiles', data=daily_quantiles[daily_quantiles['Site']=='US-Mo2'], hue='Site', fit_reg=False)
plt.show()

# Checking the above to see if its a certain month or season we should be worried about
KFS_quantiles = daily_quantiles[daily_quantiles['Site']=='US-KFS']
KFS_quantiles.month_day = pd.to_datetime(KFS_quantiles.month_day,format='%m-%d')
fig, ax = plt.subplots()
plt.scatter(KFS_quantiles.month_day, KFS_quantiles.AMF_quantiles,c='red',s=.5)
plt.scatter(KFS_quantiles.month_day, KFS_quantiles.PRISM_quantiles,c='blue',s=.5)
plt.show()

Mo2_quantiles = daily_quantiles[daily_quantiles['Site']=='US-Mo2']
Mo2_quantiles.month_day = pd.to_datetime(Mo2_quantiles.month_day,format='%m-%d')
fig, ax = plt.subplots()
plt.scatter(Mo2_quantiles.month_day, Mo2_quantiles.AMF_quantiles,c='red',s=.5)
plt.scatter(Mo2_quantiles.month_day, Mo2_quantiles.PRISM_quantiles,c='blue',s=.5)
plt.show()





   