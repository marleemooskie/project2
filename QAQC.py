'''
In this script, I will load the data and attempt to start fitting my first RNN
or ESN to it. I will also investigate the days surrounding a heatwave.

'''

import os
os.chdir("/Users/marleeyork/Documents/project2")
from load_data import *
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',300)
pd.set_option('display.max_rows',100)

# Loading in the AmeriFlux data across all sites with soil water content
# This automatically loads selected for columns, defined in loadAMF
df = loadAMF(path='/Users/marleeyork/Documents/project2/data/AMFdataDD',
                 measures=['TIMESTAMP','TA_F','SW_IN_F','VPD_F','P_F','NEE_VUT_REF','RECO_NT_VUT_REF','GPP_NT_VUT_REF','SWC_F_MDS_1'])

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
   