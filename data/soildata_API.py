'''
This script will request soil data from the soil grid restAPI.
'''
import requests
import time

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


if __name__ == "__main__":

    # Example input: a dictionary of sites with (lat, lon)
    sites = {
        "Tower_A": (45.5, -110.2),
        "Tower_B": (53.31, -105.8),
        "Tower_C": (39.02, -120.23)
    }

    soil_results = fetch_soilgrids_for_sites(
        site_dict=sites,
        variables=["sand", "clay", "silt"],
        depth_intervals=["0-5", "5-15", "15-30"]
    )

    # Print the result for one site
    import json
    print(json.dumps(soil_results["Tower_A"], indent=2))
