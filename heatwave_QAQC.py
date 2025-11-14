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

