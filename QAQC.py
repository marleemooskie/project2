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
from heatwave_QAQC import *
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

# Loading in historical mean data from PRISM
# Need to include Canada data with this eventually
historical_tmean = pd.read_csv("/Users/marleeyork/Documents/project2/data/PRISM/extracted_daily_tmean.csv")

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
##                        Bad Data Edits                                     ##
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

# Whs days before 2016
drop_list.extend(df[(df['Site']=='US-Whs') & (df['TIMESTAMP'] > "2016-01-01")].index)

# Ho1 days before 2008
drop_list.extend(df[(df['Site']=='US-Ho1') & (df['TIMESTAMP'] < "2008-01-01")].index)

# These are corrections made to cropland (CRO) sites, which have been removed
# Mo1 days before 2016-01-01 (drop_list.len == 2978)
# drop_list.extend(df[(df['Site'] == 'US-Mo1') & (df['TIMESTAMP'] < '2016-01-01')].index)

# Drop the drop list! (df.shape == (159306,10))
df = df.drop(index=drop_list,errors='ignore')

# Eventually I need to write this to a dataframe, but I'm not sure whats going on 
# with some of my sites for now.

###############################################################################
##                        Replacing SWC Edits                                ##
###############################################################################
'''
The following code selects sites that have bad SWC_F_MDS_1 data and replaces it
with another depth of SWC. A categorical variables "SWC_depth" is added that indicates
the depth of SWC value we are using for a given site. So far, this has only been
useful for site US-Ho1.

At the end of the day, the data for Syv and Kon came out even worse doing this,
so they are going to be removed. If we need to do this with other sites in the future,
then we can go ahead and use this code framework to reproduce integrating different
SWC variables.
'''
# Check what SWC variables these three files have
swc_measures = ["SWC_F_MDS_1","SWC_F_MDS_2","SWC_F_MDS_3","SWC_F_MDS_4","SWC_F_MDS_5",
                "SWC_F_MDS_1_QC","SWC_F_MDS_2_QC","SWC_F_MDS_3_QC","SWC_F_MDS_4_QC",
                "SWC_F_MDS_5_QC"]

shared_swc = find_shared_variables('/Users/marleeyork/Documents/project2/data/AMFdataDD',swc_measures)
site_presence = shared_swc['site_presence']
site_presence[site_presence['Site'].isin(['US-Ho1','US-Syv','US-Kon'])]

# Find filepaths for for US-Ho1, US-Syv, and US-Kon
Ho1_path = "/Users/marleeyork/Documents/project2/data/AMFdataDD/AMF_US-Ho1_FLUXNET_SUBSET_DD_1996-2023_3-6.csv"
Kon_path = "/Users/marleeyork/Documents/project2/data/AMFdataDD/AMF_US-Kon_FLUXNET_SUBSET_DD_2004-2019_5-7.csv"
Syv_path = "/Users/marleeyork/Documents/project2/data/AMFdataDD/AMF_US-Syv_FLUXNET_SUBSET_DD_2001-2023_4-6.csv"

# Load in the dataset with all SWC variables of interest (starting with SWC_F_MDS_2 for now)
Ho1_SWC = loadAMFFile(this_file=Ho1_path, measures=["TIMESTAMP","SWC_F_MDS_2"])
Kon_SWC = loadAMFFile(this_file=Kon_path,measures=["TIMESTAMP","SWC_F_MDS_2"])
Syv_SWC = loadAMFFile(this_file=Syv_path,measures=["TIMESTAMP","SWC_F_MDS_3"])

# Add site label to each
Ho1_SWC['Site'] = ['US-Ho1'] * Ho1_SWC.shape[0]
Kon_SWC['Site'] = ['US-Kon'] * Kon_SWC.shape[0]
Syv_SWC['Site'] = ['US-Syv'] * Syv_SWC.shape[0]

# Concatenate these into one dataframe
swc2_data = pd.concat([Ho1_SWC,Kon_SWC])
swc3_data = Syv_SWC

# Create two other dataframes for sites with the other two depths of soil water data
df_swc = df[df['Site'].isin(['US-Ho1','US-Kon'])]
df_swc_Syv = df[df['Site'].isin(['US-Syv'])]

# Merge the two dataframes together
df_swc = pd.merge(df_swc,swc2_data,on=['Site','TIMESTAMP'],how='inner')
df_swc_Syv = pd.merge(df_swc_Syv,swc3_data,on=['Site','TIMESTAMP'],how='inner')

# Adding a categorical variable that is soil water depth we are going to use for that site
df_swc['SWC_depth'] = ['2'] * df_swc.shape[0]
df_swc_Syv['SWC_depth'] = ['3'] * df_swc_Syv.shape[0]

# Remove these 3 sites from the overall dataframe
df = df[~df['Site'].isin(['US-Ho1','US-Kon','US-Syv'])]

# Add a soil water depth identifier
df['SWC_depth'] = ['1'] * df.shape[0]

# Remove SWC_F_MDS_1 from the 3 problem sites df
df_swc = df_swc.drop(columns=['SWC_F_MDS_1'])
df_swc_Syv = df_swc_Syv.drop(columns=['SWC_F_MDS_1'])

# Rename the SWC variables to a neutral name
df = df.rename(columns={'SWC_F_MDS_1':'SWC'})
df_swc = df_swc.rename(columns={'SWC_F_MDS_2': 'SWC'})
df_swc_Syv = df_swc_Syv.rename(columns={'SWC_F_MDS_3':'SWC'})

# Check that all the columns align
df_swc.columns.isin(df.columns)
df.columns.isin(df_swc.columns)
df_swc.columns.isin(df_swc_Syv.columns)
df.columns.isin(df_swc_Syv.columns)
df_swc_Syv.columns.isin(df.columns)
df_swc_Syv.columns.isin(df_swc.columns)

# Concatenate the two dataframes
df = pd.concat([df,df_swc,df_swc_Syv])

# Removing Syv and Kon since these other SWC values did not help
df = df[~df['Site'].isin(['US-Kon','US-Syv'])]


###############################################################################
##                                  QAQC                                     ##
###############################################################################
'''
The following code will plot timeseries of important variables to investigate
any issues in the data, or sites that need to be dropped.
'''
vars_to_plot = [
    "TA_F",
    "SW_IN_F",
    "VPD_F",
    "P_F",
    "SWC",
    "GPP_NT_VUT_REF"
]

for site in df.Site.unique():
    flux_data = df[(df['Site']==site)]
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
##                         TEMPERATURE QC CHECK                              ##
###############################################################################
'''
The following code will investigate the quality of temperature data, especially
that of the highest 95th percentile for each day.
'''
# This confirms that the QAQC for daily temperature is available at all our sites
shared_tempQAQC = find_shared_variables('/Users/marleeyork/Documents/project2/data/AMFdataDD',measures=['TA_F_QC'])
print(shared_tempQAQC['available_variables'])

# This confirms that the QAQC for hourly temperature is available at all our sites
shared_tempQAQC_hourly = find_shared_variables('/Users/marleeyork/Documents/project2/data/AMFdata_HH',measures=['TA_F_QC'])
print(shared_tempQAQC_hourly['available_variables'])

# Download the daily temperature and QAQC variables
ta = loadAMF(path = "/Users/marleeyork/Documents/project2/data/AMFdataDD",measures=['TIMESTAMP','TA_F','TA_F_QC'])
ta_H = loadAMF(path = "/Users/marleeyork/Documents/project2/data/AMFdata_HH",measures=['TIMESTAMP_START','TA_F','TA_F_QC'])

# Loading in the heatwaves so that I may see if any of them are defined by these low QAQC days
os.chdir("/Users/marleeyork/Documents/project2/heatwave_definition/")
from testing_heatwaves import *

# Remove any observations not in df
df_obs = df[['Site','TIMESTAMP']]
ta = pd.merge(df_obs,ta,on=['Site','TIMESTAMP'],how='inner')

# Looking at the observations with a quality control flag
QAQC_counts = ta.groupby('Site')['TA_F_QC'].apply(lambda x: (x < .5).sum())
print(QAQC_counts)

QAQC_counts_hourly = ta_H.groupby(['Site'])['TA_F_QC'].apply(lambda x: (x < .5).sum())
print(QAQC_counts_hourly)

# Adding a month variable to see if there a certain time of the year that is an issue
ta['Month'] = ta.TIMESTAMP.dt.month
ta_H['Month'] = ta_H.TIMESTAMP_START.dt.month

# Grouping again, but this time by site and month
QAQC_counts = ta.groupby(['Site','Month'])['TA_F_QC'].apply(lambda x: (x < .5).sum())
with pd.option_context('display.max_rows', None):
    print(QAQC_counts)

QAQC_counts_hourly = ta_H.groupby(['Site','Month'])['TA_F_QC'].apply(lambda x: (x < .5).sum())
with pd.option_context('display.max_rows', None):
    print(QAQC_counts_hourly)


# Create label of quality control
TA_QAQC = []
for value in ta.TA_F_QC:
    if (value < .5):
        TA_QAQC.append(1)
    else:
        TA_QAQC.append(0)
        
ta['TA_flag'] = TA_QAQC

# Plotting temperature timeseries with daily QAQC flag to see if they are continuous
import matplotlib.pyplot as plt

colors = {'0': 'blue', '1':'red'}

for site in ta.Site.unique():
    flux_data = ta[(ta['Site']==site)]
    flux_data['TA_flag'] = flux_data['TA_flag'].astype("str")
    fig, ax = plt.subplots()
    
    plt.scatter(flux_data.TIMESTAMP,flux_data.TA_F,c=flux_data.TA_flag.map(colors),s=.5)
    plt.title(site)

    plt.tight_layout()
    plt.show()
    
    input("Press [Enter] to continue...")

# Comparing low QAQC temperature days with PRISM temperature
# Merge PRISM data with daily temperature data
tmean_long = pd.melt(historical_tmean,
                     id_vars='date',
                     var_name = 'Site',
                     value_name = 'Tmean_PRISM'
                     )

tmean_long.date = pd.to_datetime(tmean_long.date)
tmean_long.columns = ['TIMESTAMP','Site','Tmean_PRISM']

# Merge AmeriFlux data with PRISM data
ta = pd.merge(ta,tmean_long,on=['Site','TIMESTAMP'],how='inner')

# Unpack hotdays from heatwaves_EHF
# Since this QAQC is looking at average daily temperature, I am only starting
# with the EHF heatwaves
heatwave_indicator_EHF = pd.DataFrame(columns=['date','heatwave_indicator','Site'])
for site in ta.Site.unique():
    site_heatwaves = heatwaves_EHF[site]['indicator']
    site_heatwaves['Site'] = [site] * site_heatwaves.shape[0]
    heatwave_indicator_EHF = pd.concat([heatwave_indicator_EHF,site_heatwaves])
heatwave_indicator_EHF.columns = ['TIMESTAMP','heatwave_indicator','Site']
# Merge heatwave indicator onto the temperature QAQC data
ta = pd.merge(ta,heatwave_indicator_EHF,on=['Site','TIMESTAMP'])

# Plotting QAQC flagged days by site
# This is excluding any Canada and Alaska sites for now
flagged_ta = ta[ta['TA_flag']==1]
flagged_ta = flagged_ta[flagged_ta['Tmean_PRISM']!=-9999]

site_cat = flagged_ta['Site'].unique()
colors = plt.cm.tab10(range(len(site_cat))) 
color_map = dict(zip(site_cat, colors))

fig, ax = plt.subplots()
for site in site_cat:
    subset = flagged_ta[flagged_ta['Site'] == site]
    plt.scatter(subset['TA_F'], subset['Tmean_PRISM'], color=color_map[site], label=site,s=.7,alpha=.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("AMF Daily Avg Temperature")
plt.ylabel("PRISM Daily Avg Temperature")
plt.show()

# Plotting QAQC flagged days by whether or not they are in a heatwave
heatwave_cat = flagged_ta['heatwave_indicator'].unique()
colors_heatwave = plt.cm.tab10(range(len(heatwave_cat)))
color_heatwave_map = dict(zip(heatwave_cat,colors_heatwave))

fig, ax = plt.subplots()
for presence in heatwave_cat:
    subset = flagged_ta[flagged_ta['heatwave_indicator'] == presence]
    plt.scatter(subset['TA_F'], subset['Tmean_PRISM'], color = color_heatwave_map[presence], label=presence, s=.7, alpha=.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel("AMF Daily Avg Temperature")
plt.ylabel("PRISM Daily Avg Temperature")
plt.show()

# Finding those heatwaves that are invalid due to too much low quality temperature data
heatwave_QAQC = pd.DataFrame(columns=['start_date','end_date','QAQC_percentage','heatwave_invalidity','Site'])
for site in heatwaves_EHF.keys():
    print(site)
    site_qaqc = avg_QAQC_check(site_heatwave_dictionary = heatwaves_EHF[site],
                               dates = ta[ta['Site']==site].TIMESTAMP,
                               TA_QAQC = ta[ta['Site']==site].TA_F_QC,
                               QAQC_threshold = .5,
                               heatwave_threshold = .75
                               )
    site_qaqc['Site'] = [site] * site_qaqc.shape[0]
    heatwave_QAQC = pd.concat([heatwave_QAQC,site_qaqc])
    
    
    

# Running QAQC heatwave check to explore heatwaves that do not pass quality checks
###############################################################################
##                   MAXIMUM TEMPERATURE CHECKS                              ##
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
sb.lmplot(x='AMF_quantiles', y='PRISM_quantiles', data=daily_quantiles, hue='Site', fit_reg=False)
plt.show()

# Checking the above to see if its a certain month or season we should be worried about
# It is consistently these October/November 95th quantile temperatures that
# are very different.
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






   