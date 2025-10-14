# This script includes all the functions to load in your data
# Loading packages
import pandas as pdta
from datetime import datetime
import os

# Name: loadAMF()
# Summary: Reads a directory of AmeriFlux files into one dataframe

# Input: path ~ filepath for AmeriFlux data directory
#        measures ~ list of variables we want to pull from each site, not
#        not including site itself (that will be done for you)

# Output: AMF_data ~ merged dataframe of all site data with a site identifier added

def loadAMF(path, measures=['TIMESTAMP_START','TA_F','SW_IN_F','VPD_F','P_F',
                            'NEE_VUT_REF','RECO_NT_VUT_REF','GPP_NT_VUT_REF']):
    # Check if the filepath is actually a directory or not
    try:
        os.scandir(path) 
    except FileNotFoundError:
        print("Thats not a valid filepath, check for error.")

    except NotADirectoryError:
        # it wasn't a directory that was passed in
        # but this doesn't yet test if the file exists, fix that!
        print("Path for a single file was input, please provide a directory.")
        
    else:
        # it was a directory that was passed in, so let's make use of it
        print('Directory name passed in')
        
        # If given a successful directory...
        # Add site to the column list
        measures.append('Site')
        my_columns = measures
        # Pull all the filepaths within the directory
        paths = [f.path for f in os.scandir(path) if (f.is_file() and f.name != '.DS_Store')]
        
        AMF_data = pd.DataFrame(columns=my_columns)
        # Loop through each file in the path
        for this_file in paths:
            # Retrieve an AmeriFlux dataframe
            data =  loadAMFFile(this_file,measures)
            # Concatenate the data
            AMF_data = pd.concat([AMF_data,data],ignore_index=True)
    
    return AMF_data

# Name: loadAMFFile()
# Summary: Reads a singular AmeriFlux file into a dataframe

# Input: this_file ~ filepath for AMF data saved as csv
#        measures ~ list of variables that will be included in data

# Output: file_df ~ AMF data organized into dataframe with additional site column

def loadAMFFile(this_file, measures):
    # Pull the site name from the filename
    filename = os.path.basename(this_file)
    site = filename[4:10]
    print("Loading site... " + site)
    # Read the csv
    file_df = pd.read_csv(this_file)
    # Convert the timestamp into a datetime object
    file_df.TIMESTAMP_START = pd.to_datetime(file_df.TIMESTAMP_START, format='%Y%m%d%H%M')
    # Add in the site oclumn
    file_df.loc[:,'Site'] = [site]*len(file_df)
    # Select the columns that you want
    file_df = file_df[measures]
    
    return file_df






