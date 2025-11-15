'''
This script will request soil data from the soil grid restAPI.
'''
import requests
import time
import pickle

def fetch_soilgrids_point(lon, lat, variables=None, depth_intervals=None,
                          endpoint="https://rest.isric.org/soilgrids/v2.0/properties/query"):
    """
    Fetch SoilGrids property values at a point.
    """
    if variables is None:
        variables = ["sand", "clay", "silt"]
    if depth_intervals is None:
        depth_intervals = ["0-5", "5-15", "15-30"]

    params = {
        "lon": lon,
        "lat": lat,
        "variables": ",".join(variables),
        "depth_intervals": ",".join(depth_intervals)
    }

    response = requests.get(endpoint, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_soilgrids_for_sites(site_dict, variables=None, depth_intervals=None):
    """
    Loop through multiple sites and store SoilGrids results in a dictionary.

    Parameters
    ----------
    site_dict : dict
        Format: { "SiteName": (lat, lon), ... }
    variables : list
        Soil variables to query
    depth_intervals : list
        Depth intervals

    Returns
    -------
    dict
        { "SiteName": response_json }
    """
    results = {}

    for site, (lat, lon) in site_dict.items():
        print(f"Requesting SoilGrids for {site} at (lat={lat}, lon={lon})...")
        try:
            data = fetch_soilgrids_point(
                lon=lon,
                lat=lat,
                variables=variables,
                depth_intervals=depth_intervals
            )
            results[site] = data
            print(f"✓ Success: {site}")

        except Exception as e:
            print(f"✗ Error fetching data for {site}: {e}")
            print("  Retrying in 5 seconds...")
            time.sleep(5)
            try:
                data = fetch_soilgrids_point(
                    lon=lon,
                    lat=lat,
                    variables=variables,
                    depth_intervals=depth_intervals
                )
                results[site] = data
                print(f"✓ Success on retry: {site}")
            except Exception as e2:
                print(f"✗ Failed again for {site}. Skipping.")
                results[site] = None

    return results

# Load in my site locations
site_loc = pd.read_csv("/Users/marleeyork/Documents/project2/data/site_locations_for_PRISM.csv")

if __name__ == "__main__":

    # Initialize a dictionary for site location
    sites = {}
    
    for site in site_loc.Site.unique():
        sites[site] = (site_loc[site_loc['Site']==site].Lat,site_loc[site_loc['Site']==site].Lon)
        
    
    soil_results = fetch_soilgrids_for_sites(
        site_dict=sites,
        variables=["sand", "clay", "silt"],
        depth_intervals=["0-5", "5-15", "15-30"]
    )

    # Print the result for one site
    import json
    print(json.dumps(soil_results["US-WCr"], indent=2))
    
    
# Saving data into a pickle
with open("soil_data.pickle", "wb") as file:
    pickle.dump(soil_results, file)
    
# Unpacking the dictionary into a dataframe of the variables I am interested in
soil_results['CA-Cbo'].keys()
soil_results['CA-Cbo']['type']
soil_results['CA-Cbo']['geometry']
soil_results['CA-Cbo']['properties']['layers'][0]['name']
soil_results['CA-Cbo']['query_time_s']

# Unpacking all available variables
soil_var = []
for i in range(len(soil_results['CA-Cbo']['properties']['layers'])):
    soil_var.append(soil_results['CA-Cbo']['properties']['layers'][i]['name'])

soil_var.insert(0,'Depth')
soil_var.insert(0,'Site')


# Initialize dataframe with available soil variable names
soil_df = pd.DataFrame(columns=soil_var)

# Loop through each soil depths
for site in soil_results.keys():
    # Unpacking the first soil depth
    all_values = [site]

    # Get the depth
    all_values.append(soil_results[site]['properties']['layers'][i]['depths'][0]['label'])
    
    # Loop through each variable and add value to the list
    for i in range(len(soil_var)-2):
        print(j)
        print(i)
        print(soil_results[site]['properties']['layers'][i]['name'])
        all_values.append(soil_results[site]['properties']['layers'][i]['depths'][0]['values']['mean'])

    # Add this to the dataframe
    soil_df.loc[len(soil_df)] = all_values

# Save this dataframe
# In the future, I want to add other soil depths, but its being weird for now
# I think its because theres different number of variables for each depth
soil_df.to_csv("soil_df.csv",index=False)







