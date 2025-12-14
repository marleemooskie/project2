'''
This script will calculate magitude and intensities of each heatwave.
'''
import pandas as pd

def cumulative_exceedence(heatwaves_dictionary, daily_AMF_TA, historical_data):
    '''
    This function integrates over a certain variable during a heatwave and takes
    the average of it.
    
    Parameters
    ----------
    heatwaves_dictionary : TYPE
        DESCRIPTION. Dictionary of all heatwaves.
    
    
    historical_dates : TYPE
        DESCRIPTION. list of all heatwaves
    
    historical_temperatures : TYPE
        DESCRIPTION. list of all dates corresponding to historical data
    
    Returns
    -------
    heatwave_categories : TYPE
        DESCRIPTION. A categorical list for the type of overlapping heatwave each 
        day is, including None, Night, Day, Overall, Night-intensified, 
        Day-intensified, Day-Night Spike, and Triad

    '''
    
    for site in heatwaves_dictionary.keys():
        # Pull out the dictionary for that site
        site_dictionary = heatwaves_dictionary[site]
        site_temperature = daily_AMF_TA[daily_AMF_TA['Site']==site]
        site_temperature['date'] = pd.to_datetime(site_temperature['date'])
        site_historical = historical_data[historical_data['Site']==site]
        historical_temperature = site_historical['hist_TA']
        historical_dates = pd.to_datetime(site_historical['date'])
        site_heatwaves = pd.DataFrame({'start_dates':site_dictionary['start_dates'],
                                       'end_dates':site_dictionary['end_dates']})
        # Calculate the 3 day moving average for historical data
        historical_DMT = []
        for i in range(2,len(historical_temperature)):
            DMT = historical_temperature[(i-2):(i+1)].sum() / 3
            historical_DMT.append(DMT)
        # Organize into dataframe
        historical_DMT_df = pd.DataFrame({"date":historical_dates[2:],
                                          "DMT":historical_DMT})
        # Add a month day column
        historical_DMT_df['month_day'] = historical_DMT_df.date.dt.strftime('%m-%d')
        # Group by month day and calculate 95th of DMT
        T95_by_day = (
            historical_DMT_df
            .groupby("month_day")["DMT"]
            .quantile(.95)
            .reset_index(name="T95")
        )
        # Add a month_day column to site temperature (not historical)
        site_temperature['month_day'] = site_temperature.date.dt.strftime('%m-%d')
        # Merge timeseries data with the DMT 95th quantiles
        site_temperature = pd.merge(site_temperature,T95_by_day,on="month_day",how="left")
        # Calculate difference between threshold and observed temperature
        site_temperature['difference'] = site_temperature.TA_F - site_temperature.T95

        # Now loop through each heatwave
        cumulative_exceedence = []
        average_exceedence = []
        for i in range(len(site_heatwaves)):
            # Find date range of heatwaves
            start = site_heatwaves['start_dates'][i]
            end = site_heatwaves['end_dates'][i]
            date_range = pd.date_range(start,end)
            # Find the corresponding temperatures for these dates
            heatwave_difference = site_temperature[site_temperature['date'].isin(date_range)].difference
            # Replace negative values with zero
            heatwave_difference[heatwave_difference<0] = 0
            cumulative_exceedence.append(sum(heatwave_difference))
            average_exceedence.append(sum(heatwave_difference) / len(heatwave_difference))
        site_heatwaves['cumulative_exceedence'] = cumulative_exceedence
        site_heatwaves['average_exceedence']
        heatwave_dictionary[site]['cumulative_exceedence'] = site_heatwaves['cumulative_exceedence']
    
    return heatwave_dictionary
    