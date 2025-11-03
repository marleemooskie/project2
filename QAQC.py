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
                 skip=['/Users/marleeyork/Documents/project2/data/AMFdataDD/AMF_US-Ha1_FLUXNET_SUBSET_DD_1991-2020_3-5.csv',
                 '/Users/marleeyork/Documents/project2/data/AMFdataDD/AMF_US-Vcp_FLUXNET_SUBSET_DD_2007-2024_5-7.csv',
                 '/Users/marleeyork/Documents/project2/data/AMFdataDD/AMF_US-KS2_FLUXNET_SUBSET_DD_1999-2006_3-5.csv',
                 '/Users/marleeyork/Documents/project2/data/AMFdataDD/AMF_US-Wjs_FLUXNET_SUBSET_DD_2007-2024_4-7.csv',
                 '/Users/marleeyork/Documents/project2/data/AMFdataDD/AMF_US-Mpj_FLUXNET_SUBSET_DD_2008-2024_4-7.csv',
                 '/Users/marleeyork/Documents/project2/data/AMFdataDD/AMF_US-Hn3_FLUXNET_SUBSET_DD_2017-2018_5-7.csv'],
                 measures=['TIMESTAMP','TA_F','SW_IN_F','VPD_F','P_F','NEE_VUT_REF','RECO_NT_VUT_REF','GPP_NT_VUT_REF','SWC_F_MDS_1'])

df.columns
df.shape

###############################################################################
##                                 Edits                                     ##
###############################################################################

# Drop any rows with values that are -9999 (new shape is (211973,10))
mask = df.apply(lambda col: col == -9999).any(axis=1)
df = df[~mask]

# Now we will do site specific filtering
# Initializing a drop list that we will fill with indices to remove
drop_list = []

# SRG days before 05/02/2018 (drop_list.len == 70)
drop_list.extend(df[(df['Site'] == 'US-SRG') & (df['TIMESTAMP'] < '2008-05-02')].index)

# SRM days after 2017-05-01 (drop_list.len == 2871)
drop_list.extend(df[(df['Site'] == 'US-SRM') & (df['TIMESTAMP'] > '2017-05-01')].index)

# Mo1 days before 2016-01-01 (drop_list.len == 2978)
drop_list.extend(df[(df['Site'] == 'US-Mo1') & (df['TIMESTAMP'] < '2016-01-01')].index)

# LP1 days after 2016-12-01 (drop_list.len == 4104)
drop_list.extend(df[(df['Site'] == 'CA-LP1') & (df['TIMESTAMP'] > '2016-12-01')].index)

# Me2 days after 2021-10-01 (drop_list.len == 4402)
drop_list.extend(df[(df['Site'] == 'US-Me2') & (df['TIMESTAMP'] > '2021-10-01')].index)

# BZS days prior to 2015-01-01 (drop_list.len == 4689)
drop_list.extend(df[(df['Site'] == 'US-BZS') & (df['TIMESTAMP'] < '2015-01-01')].index)

# Syv days prior to 2012-01-01 (drop_list.len == 7033)
drop_list.extend(df[(df['Site'] == 'US-Syv') & (df['TIMESTAMP'] < '2012-01-01')].index)

# KFS days prior to 2009-01-01 (drop_list.len == 7399)
drop_list.extend(df[(df['Site'] == 'US-KFS') & (df['TIMESTAMP'] < '2009-01-01')].index)

# Drop the drop list! (df.shape == (204573,10))
df = df.drop(drop_list)

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


    plt.tight_layout()
    plt.show()
    
    input("Press [Enter] to continue...")
   