'''
This script includes functions that check the quality of heatwaves based on 
the AmeriFlux QAQC temperature flags.
'''
import pandas as pd

def avg_QAQC_check(site_heatwave_dictionary, dates, TA_QAQC, QAQC_threshold,
                   heatwave_threshold):
    '''
    Description
    -----------
    This function identifies heatwaves that are invalid due to having too high
    of a percentage of low quality AmeriFlux data.
    
    Parameters
    ----------
    site_heatwave_dictionary : TYPE
        DESCRIPTION. Dictionary provided by fit_heatwaves with method="EHF" for
        one site. E.g. site_heatwave_dictionary = heatwaves_EHF['US-GLE']
    dates : TYPE
        DESCRIPTION. Dates for AmeriFlux data associated with the following QAQC values.
    TA_QAQC : TYPE
        DESCRIPTION. TA_F_QAQC values associated with the above dates for one given site.
    QAQC_threshold : TYPE
        DESCRIPTION. The bottom TA_F_QAQC threshold that determines a day of inacceptable data.
    heatwave_threshold : TYPE
        DESCRIPTION. The percentage of inacceptable data days a heatwave can have and 
        still be considered a valid heatwave.
    Returns
    -------
    heatwave_qaqc : TYPE
        DESCRIPTION. DataFrame of start_date, end_date, percentage of days in the 
        heatwave that have a QAQC below the necessary threshold (QAQC_percentage), and validity of the
        heatwave based on accepted QAQC_percentage as defined by heatwave threshold (heatwave_invalidity)
    '''
    start_dates = site_heatwave_dictionary['start_dates']
    end_dates = site_heatwave_dictionary['end_dates']
    ta_qaqc = pd.DataFrame({'dates':dates,'QAQC':TA_QAQC})
    heatwave_qaqc = pd.DataFrame(columns=['start_date','end_date','QAQC_percentage','heatwave_invalidity'])
    # Loop through each heatwave
    for start, end in zip(start_dates,end_dates):
        # Create a range of dates between the start and end
        date_range = pd.date_range(start=start, end=end)
        # Find the QAQC values in these dates
        heatwave_QAQC_values = ta_qaqc[ta_qaqc['dates'].isin(date_range)]
        # Determine if they are flagged as being below the threshold
        QAQC_flag = []
        for quality in heatwave_QAQC_values.QAQC:
            if (quality < QAQC_threshold):
                QAQC_flag.append(1)
            else:
                QAQC_flag.append(0)
        
        if len(QAQC_flag) == 0:
            continue
        else:
            # Find percentage of flagged days
            QAQC_percentage = sum(QAQC_flag) / len(QAQC_flag)
            invalidity_flag = 0 if (QAQC_percentage < heatwave_threshold) else 1
            # Add start date, end date, percentage of bad data days, and heatwave validity to dataframe
            this_site = pd.DataFrame({'start_date':[start],
                                  'end_date':[end],
                                  'QAQC_percentage':[QAQC_percentage],
                                  'heatwave_invalidity':[invalidity_flag]})
            # Concatenate with QAQC of other heatwaves
            heatwave_qaqc = pd.concat([heatwave_qaqc,this_site])
    return heatwave_qaqc

def minmax_QAQC_check(site_heatwave_dictionary, dates, TA, TA_QAQC, heatwave_threshold):
    '''
    Description
    -----------
    This function identifies heatwaves that are invalid due to having too high
    of a percentage of low quality AmeriFlux data. This is for those heatwaves defined by
    hourly data, like the min/max approaches.
    
    I don't think I actually need to use this, but TBD. Could still do an hourly thing
    where if the max/min temperature is bad, it defaults to the next temperature.
    Our min/max temperatures were really well correlated with PRISM data though.
    
    Parameters
    ----------
    site_heatwave_dictionary : TYPE
        DESCRIPTION. Dictionary provided by fit_heatwaves with method="EHF" for
        one site. E.g. site_heatwave_dictionary = heatwaves_EHF['US-GLE']
    dates : TYPE
        DESCRIPTION. Dates for AmeriFlux data associated with the following QAQC values.
    TA_QAQC : TYPE
        DESCRIPTION. TA_F_QAQC values associated with the above dates for one given site.
    QAQC_threshold : TYPE
        DESCRIPTION. The bottom TA_F_QAQC threshold that determines a day of inacceptable data.
    heatwave_threshold : TYPE
        DESCRIPTION. The percentage of inacceptable data days a heatwave can have and 
        still be considered a valid heatwave.
    Returns
    -------
    heatwave_qaqc : TYPE
        DESCRIPTION. DataFrame of start_date, end_date, percentage of days in the 
        heatwave that have a QAQC below the necessary threshold (QAQC_percentage), and validity of the
        heatwave based on accepted QAQC_percentage as defined by heatwave threshold (heatwave_invalidity)
    '''
    # Initialize list to hold flags for heatwaves surpassing the valid amount of downscaled data
    flag = []
    # Get find max hourly temperature
    hourly_TA = pd.DataFrame({'dates':dates,'TA':TA,'TA_QAQC':TA_QAQC})
    hourly_TA['dates_dt'] = pd.to_datetime(hourly_TA['dates'].dt.date)
    
    # Loop through each heatwave
    df = site_heatwave_dictionary['summary']
    for i in range(df.shape[0]):
        # Isolate a heatwave
        this_heatwave = df.iloc[0]
        # Retrieve the dates for a given heatwave
        this_heatwave_dates = pd.date_range(this_heatwave.start_dates, this_heatwave.end_dates)
        # Find temperatures and QAQC for this heatwave
        this_heatwave_hourly = hourly_TA[hourly_TA['dates_dt'].isin(this_heatwave_dates)]
        # Get the max temperature for the heatwave days
        max_temperatures = find_max_temperatures(this_heatwave_hourly.dates, this_heatwave_hourly.TA)
    
        downscaled = []
        # Get the QAQC associated with that temperature
        for j in range(max_temperatures.shape[0]):
            # Isolate the date we are looking at
            this_date = max_temperatures.iloc[j]['date']
            this_date = pd.to_datetime(pd.to_datetime(this_date).date())
            # Isolate what was found as the maximum temperature for that day
            this_temperature = max_temperatures.iloc[j]['max_temperature']
            # Find the hour of the day that the maximum temperature came from
            max_hour = this_heatwave_hourly[(this_heatwave_hourly['dates_dt']==this_date) 
                                            & (this_heatwave_hourly['TA']==this_temperature)]
            # Check the QAQC for that day
            qaqc = list(max_hour['TA_QAQC'])
            # Check if it is gap-filled
            if (qaqc[0]==2):
                downscaled.append(1)
            else:
                downscaled.append(0)
    
        # Calculate the percentage of the heatwave that is based on downscaled data
        heatwave_percentage = sum(downscaled) / len(downscaled)
        fail = 0 if (heatwave_percentage < heatwave_threshold) else 1
        flag.append(fail)
    # Merge this onto the heatwave summary
    df['QAQC_flag'] = flag
    # If it is bad, then flag it as bad
    heatwave_qaqc = df
    
    return heatwave_qaqc


