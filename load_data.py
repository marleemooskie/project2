# This script includes all the functions to load in your data
# Loading packages
import pandas as pd
from datetime import datetime
import os

def loadAMF(path, skip = '',measures=['TIMESTAMP','TA_F','SW_IN_F','VPD_F','P_F',
                            'NEE_VUT_REF','RECO_NT_VUT_REF','GPP_NT_VUT_REF']):
    '''
    Name: loadAMF()
    Summary: Reads a directory of AmeriFlux files into one dataframe

    Input: path ~ filepath for AmeriFlux data directory
           measures ~ list of variables we want to pull from each site, not
           not including site itself (that will be done for you)

    # Output: AMF_data ~ merged dataframe of all site data with a site identifier added
    '''
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
            if (this_file==skip):
                continue
            else:
                # Retrieve an AmeriFlux dataframe
                data =  loadAMFFile(this_file,measures)
                # Concatenate the data
                AMF_data = pd.concat([AMF_data,data],ignore_index=True)
    
    return AMF_data



def loadAMFFile(this_file, measures):
    '''
    Name: loadAMFFile()
    Summary: Reads a singular AmeriFlux file into a dataframe

    Input: this_file ~ filepath for AMF data saved as csv
           measures ~ list of variables that will be included in data

    Output: file_df ~ AMF data organized into dataframe with additional site column
    '''
    # Pull the site name from the filename
    filename = os.path.basename(this_file)
    site = filename[4:10]
    print("Loading site... " + site)
    # Pull out the resolution
    resolution = filename[26]
    print(resolution)
    # Read the csv
    file_df = pd.read_csv(this_file)
    # Convert the timestamp into a datetime object
    if (resolution == "H"):
        file_df.TIMESTAMP_START = pd.to_datetime(file_df.TIMESTAMP_START, format='%Y%m%d%H%M')
    elif (resolution == "D"):
        file_df.TIMESTAMP = pd.to_datetime(file_df.TIMESTAMP, format='%Y%m%d')
    else:
        print("Resolution error for " + site + ", check the timestamp.")
    # Add in the site oclumn
    file_df.loc[:,'Site'] = [site]*len(file_df)
    # Select the columns that you want
    file_df = file_df[measures]
    
    return file_df


print("All data loading functions loaded.")

# now i need to make a function that loops through each file in a directory and 
# returns the variables that they all share
def find_shared_variables(path,measures):
    
    '''
    Name: find_shared_variables()
    Summary: This returns information on variables shared across all AmeriFlux
            files from a given directory. The main purpose is to identify moisture
            variables that can be studied when looking at many sites.

    Input: path ~ filepath for directory of AmeriFlux datasets
           measures ~ the variables you want if the files have

    Output: variable_info ~ dictionary that includes the following information
                site_presence ~ 0/1 indicator of presence of each variable at each site
                total_presence ~ how many sites each variable is collected at
                available_variables ~ variables available at all sites
                unavailable_variables ~ variables not available at all sites
    '''
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
        measures.insert(0,'Site')
        my_columns = measures
        # Pull all the filepaths within the directory
        paths = [f.path for f in os.scandir(path) if (f.is_file() and f.name != '.DS_Store')]
        
        merged_data = pd.DataFrame(columns=my_columns)
        # Loop through each file in the path
        for this_file in paths:
            # Retrieve an AmeriFlux dataframe
            data =  check_for_variable(this_file,measures)
            # Concatenate the data
            merged_data = pd.concat([merged_data,data],ignore_index=True)
    
    # Sum how many sites each variable is available at
    presence_counts = merged_data.sum()
    presence_counts = presence_counts.drop('Site')
    # Print some review statements about site counts
    
    # Store all this information into a dictionary so that it is readily available
    variable_info = {}
    variable_info['site_presence'] = merged_data
    variable_info['total_presence'] = presence_counts
    
    # Make a list of variables that are available at all sites 
    available_variables = []
    unavailable_variables = []
    # Print some information on variable availability
    for i in range(len(presence_counts)):
        if (presence_counts[i] == len(merged_data)):
            available_variables.append(presence_counts.index[i])
        else:
            unavailable_variables.append(presence_counts.index[i])
            
    variable_info['available_variables'] = available_variables
    variable_info['unavailable_variables'] = unavailable_variables
    
    # Print some statements about what variables you can use
    print('You have ' + str(len(available_variables)) + ' variables shared across all sites.')
    
    if (len(available_variables) > 0):
        print('Variables available at all sites: ' + available_variables)
    
    return variable_info
    

def check_for_variable(this_file,measures):
    '''
    Name: check_for_variable()
    Summary: Checks which variables of a list exist in a file

    Input: this_file ~ the path of the file you are checking
           measures ~ the variables you want to check if the file has

    Output: measure_df ~ a pandas dataframe with the site and presence/absence
            for each variable
    
    '''
    
    # Pull the site name
    filename = os.path.basename(this_file)
    site = filename[4:10]
    # Open the file
    file = open(this_file,'r')
    # Read read the first line
    first_line = file.readline()
    # Extract all the columns
    file_columns = first_line.strip('\n').split(',')
    # Close the file
    file.close()
    # Create a dataframe with the measures as the column names
    measure_df = pd.DataFrame(columns=measures)
    # For each measure we want to test, presence in the column names = 1, and 
    # absence = 0
    presence = []
    for measure in measure_df.columns:
        if (measure == 'Site'):
            presence.append(site)
        elif (measure in file_columns):
            presence.append(1)
        else:
            presence.append(0)
    
    # Add the presence absence as a row in the dataframe
    measure_df.loc[0,:] = presence
    
    return measure_df
