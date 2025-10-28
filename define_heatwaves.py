# This script will explore the different approaches for the definition of heatwaves

# Importing packages
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from matplotlib import pyplot as plt

# FUNCTIONS ###################################################################

def find_max_temperatures(date_vector, temperature_vector):
    '''
    Name: find_max_temperatures()
    Summary: These functions aggregates subdaily values of temperature (or any other
             variable) and calculates the maximum/minimum for each day

    Input: date_vector ~ Datetime stamp with at a subdaily level (e.g., hourly or 30 minute)
           temperature_vector ~ Temperatures associated with date_vector

    Output: (min/max)_temperatures ~ Pandas dataframe with column 'date' specifying
            daily dates and max_temperature specifying the summary statistic
            for that day
    '''
    # Create dataframe of timestamp and subdaily temperature
    temp_df = pd.DataFrame({'timestamp': date_vector,
                            'temperature': temperature_vector})
    
    # Perform daily aggregation, calculate max, reogranize into new dataframe
    max_temperatures = (
    temp_df
    .groupby(temp_df['timestamp'].dt.date)['temperature']
    .max()
    .reset_index()
    .rename(columns={'timestamp': 'date', 'temperature': 'max_temperature'})
    )   
    
    # Reset the date to a datetime variable
    max_temperatures['date'] = pd.to_datetime(max_temperatures['date'])
    
    return max_temperatures


def find_min_temperatures(date_vector, temperature_vector):
    '''
    Name: find_min_temperatures()
    Summary: These functions aggregates subdaily values of temperature (or any other
             variable) and calculates the maximum/minimum for each day

    Input: date_vector ~ Datetime stamp with at a subdaily level (e.g., hourly or 30 minute)
           temperature_vector ~ Temperatures associated with date_vector

    Output: (min/max)_temperatures ~ Pandas dataframe with column 'date' specifying
            daily dates and min_temperature specifying the summary statistic
            for that day
    '''
    # Create a dataframe of timestamp and subdaily temperature
    temp_df = pd.DataFrame({'timestamp': date_vector,
                            'temperature': temperature_vector})
    
    # Perform daily aggregation, calculate max, reorganize into new dataframe
    min_temperatures = (
    temp_df
    .groupby(temp_df['timestamp'].dt.date)['temperature']
    .min()
    .reset_index()
    .rename(columns={'timestamp': 'date', 'temperature': 'min_temperature'})
    )   
    
    # Reset the date to a datetime variable
    min_temperatures['date'] = pd.to_datetime(min_temperatures['date'])

    
    return min_temperatures


def moving_window_quantile(dates, measure, measure_quantile, window_length):
    '''
    Name: moving_window_quantiles()
    Summary: For a given length of moving window, calculate a quantile for
             across all historical values of a given measure (e.g., temperature, 
             max temperatures, etc) for each day of the year. Each day's quantile
             will be based on a surrounding window, such that the day is the centre
             of the window being calculated.

    Input: window_length ~ an odd number of days you want the window length to be
           dates ~ daily dates of historical data as datetime variable
           measure ~ daily measure associated with each day of dates
           quantile ~ quantile you want to calculate over the window (e.g., 90th)

    Output: window_quantiles ~ dataframe with 'day' as a datetime variable specifying
            month and day, and 'quantile' specifying the quantile of interest for 
            that day over the surrounding window.
    
    '''
    
    # Create dataframe of dates and measure
    measure_df = pd.DataFrame({'date': dates,
                               'month_day': dates.dt.strftime('%m-%d'),
                               'measure': measure})
    
    # Create range of all month and days throughout one year
    window_centre = pd.date_range(start = '1800-01-01', end = '1800-12-31', freq = 'D')
    
    # Determine the length of days backwards and forwards we need to look
    # so that our day of interest is at the centre of our window
    half_window_length = int((window_length - 1) / 2)
    
    # Subtract and add half_window_length from date to determine start and end
    # points of the window.
    window_start = window_centre - pd.to_timedelta(half_window_length, unit='d')
    window_end = window_centre + pd.to_timedelta(half_window_length, unit = 'd')
    
    # Calculate the quantiles in each window
    # Loop through the centre, start, and end of the windows
    window_quantile = []
    for centre, start, end in zip(window_centre, window_start, window_end):
        # Create a range of M-D inside the window
        window_range = pd.date_range(start=start, end=end, freq='D')
        
        # Isolate the month and day for the window range
        window_range = window_range.strftime('%m-%d')
        
        # Create indicator for days from 'dates' that fall into the window range
        window_mask = measure_df.month_day.isin(window_range)
        
        # Calculate the quantiles for the measure over all days that fall in the window
        window_quantile.append(measure_df[window_mask].measure.quantile(measure_quantile))
        
    # Create dataframe of the month-day and the associated quantile
    window_quantiles = pd.DataFrame({'month_day': window_centre.strftime('%m-%d'),
                                    'quantiles': window_quantile})
    
    return window_quantiles


def define_hotdays(timeseries_dates, timeseries_temperature, threshold_month_day, threshold, comparison = "greater"):
    '''
    Name: define_hotdays()
    Summary: Returns an indicator (0/1) whether each day is a hot day or not based on
             whether is greater than, less than, or equal to some daily threshold (e.g.,
             90th quantile of maximum temperature)    

    Input: timeseries_dates ~ dates associated with the timeseries we want to define as hot or not
           timeseries_temperature ~ temperatures associated with each day of the timeseries
           threshold ~ the daily threshold value, in order by month-day (01-01, ..., 12-31)
           comparison ~ value "greater", "less", or "equal" to threshold

    Output: hotdays ~ indicator vector (0 or 1) of length timeseries that defines
                       each day is 1 = hot day, or 0 = not hot day
    '''
    
    # Create separate timeseries and threshold dataframes
    timeseries_df = pd.DataFrame({'date':timeseries_dates,
                                  'month_day': timeseries_dates.dt.strftime("%m-%d"),
                                  'temperature':timeseries_temperature})
    threshold_df = pd.DataFrame({'month_day':threshold_month_day,
                                 'threshold':threshold})
    
    # Merge by month so we have the timeseries with its corresponding threshold
    df = pd.merge(timeseries_df, threshold_df, on="month_day", how="left")
    
    # Create a T/F mask for whether the value for a given day is hot
    if (comparison == 'greater'):
        threshold_mask = df.temperature > df.threshold
    elif (comparison == "lesser"):
        threshold_mask = df.temperature < df.threshold
    elif (comparison == "equal"):
        threshold_mask = df.temperature == df.threshold
    else:
        print("Not a valid comparison entry... enter greater, lesser, or equal")
        
    # Create a hotdays vector that is 1/0 corresponding to T/F in a dataframe
    hotdays = pd.DataFrame({'date':timeseries_dates,
                            'hotday_indicator':threshold_mask.astype(int)})
    
    return hotdays


def define_EHF_hotdays(timeseries_dates, timeseries_temperature, 
                       historical_dates, historical_temperature):
    '''
    Name: define_EHF_hotdays()
    Summary: Returns a dataframe of heatwave dates, 0/1 indicator of EHF hotday,
             and EHF score

    Input: timeseries_dates ~ daily dates associated with average temperatures
           timeseries_temperature ~ daily average temperatures over timeseries of interest
           historical_dates ~ vector of dates over historical/climatological data
           historical_average_temperature ~ this must be DAILY AVERAGES, not min/max

    Output: EHF_df ~ dataframe including the EHF score and an indicator vector 
                        (0 or 1) of length timeseries that defines
                       each day is 1 = hot day, or 0 = not hot day based on EHF
    '''
    # Calculate the 3 day moving average for historical data
    historical_DMT = []
    for i in range(2,len(historical_temperature)):
        DMT = historical_temperature[(i-2):(i+1)].sum() / 3
        historical_DMT.append(DMT)
    
    # Get the 95th quantile of climatological mean temperature
    T95 = np.quantile(historical_DMT,.9)
    
    # Initialize lists to store all the indices
    three_day_mean_list = []
    thirty_day_mean_list = []
    EHI_accl_list = []
    EHI_sig_list = []
    EHF_hotdays = []
    for i in range(29,len(timeseries_temperature)):
        # Calculate and store the 3 day mean
        three_day_mean = timeseries_temperature[(i-2):(i+1)].sum() / 3
        three_day_mean_list.append(three_day_mean)
        # Calculate and store the 30 day mean
        thirty_day_mean = timeseries_temperature[(i-29):(i+1)].sum() / 30
        thirty_day_mean_list.append(thirty_day_mean)
        # Calculate the acclimation score
        EHI_accl = three_day_mean - thirty_day_mean
        EHI_accl_list.append(EHI_accl)
        # Calculate the universal score
        EHI_sig = three_day_mean - T95
        EHI_sig_list.append(EHI_sig)
        # Use the prior to find the extreme heat factor
        EHF = EHI_sig * max(0,EHI_accl)
        EHF_hotdays.append(EHF)
        
    print("95th percentile of historical 3-day means:", T95)
    print("Range of historical 3-day means:", np.min(historical_DMT), "to", np.max(historical_DMT))
    print("EHI_sig mean:", np.mean(EHI_sig_list), "min:", np.min(EHI_sig_list), "max:", np.max(EHI_sig_list))
    print("EHI_accl mean:", np.mean(EHI_accl_list), "min:", np.min(EHI_accl_list), "max:", np.max(EHI_accl_list))

    # Create an indicator for EHF   
    hotday_indicator = [1 if x > 0 else 0 for x in EHF_hotdays]
    # Organize into a dataframe    
    EHF_df = pd.DataFrame({'date':timeseries_dates[29:],
                           'EHF_score':EHF_hotdays,
                           'hotday_indicator':hotday_indicator}).reset_index(drop=True)
    
    return EHF_df

def find_heatwaves(hotdays, dates, minimum_length, tolerance, gap_day_window):
    """
    Name: find_heatwaves()
    Summary: Identify heatwaves, allowing up to `tolerance` non-hot days per `gap_day_window`.
    Heatwave continues as long as hotdays accumulate before tolerance runs out.

    Input: dates ~ dates across the timeseries of interest
           hotdays ~ binary (0/1) indicator of hot day, received from define_hotdays()
           minimum_length ~ minimum number of contiguous hotdays to be considered heatwave
           gap_days ~ number of gap days allowed per gap_day_window
           gap_day_window ~ number of days that gap_days can fall in 

    Output: start_dates ~ vector of dates that mark the beginning of heatwaves
            end_dates ~ vector of dates that mark the end of heatwaves

    """
    active = hotdays == 1

    start_dates, end_dates = [], []
    in_heatwave = False
    start_time = None
    tolerance_left = tolerance
    hotday_count = 0
    total_days_in_window = 0  # total days (hot + not) within current gap window

    for i in range(len(active)):
        if active[i]:
            if not in_heatwave:
                # Start new heatwave
                in_heatwave = True
                start_time = dates[i]
                hotday_count = 1
                total_days_in_window = 1
                tolerance_left = tolerance
            else:
                # Continue existing heatwave
                hotday_count += 1
                total_days_in_window += 1

                # Reset tolerance if window reached
                if total_days_in_window >= gap_day_window:
                    tolerance_left = tolerance
                    total_days_in_window = 0  # start a new window
        else:
            if in_heatwave:
                total_days_in_window += 1
                if tolerance_left > 0:
                    # Allow a cool day, but don't add to hotday_count
                    tolerance_left -= 1
                else:
                    # No tolerance left → heatwave ends
                    if hotday_count >= minimum_length:
                        end_time = dates[i - 1]
                        start_dates.append(start_time)
                        end_dates.append(end_time)
                    # Reset
                    in_heatwave = False
                    hotday_count = 0
                    tolerance_left = tolerance
                    total_days_in_window = 0

    # Handle if heatwave goes till end
    if in_heatwave and hotday_count >= minimum_length:
        end_time = dates[-1]
        start_dates.append(start_time)
        end_dates.append(end_time)
        
    # Trim off any non-hotdays around the end dates
    trimmed_end_dates = []
    for start, end in zip(start_dates, end_dates):
       mask = (dates >= start) & (dates <= end)
       segment = active[mask]

       # Walk backward to find last hotday
       for j in range(len(segment) - 1, -1, -1):
           if segment.iloc[j] if hasattr(segment, "iloc") else segment[j]:
               if hasattr(dates, "iloc"):
                   trimmed_end_dates.append(dates[mask].iloc[j])
               else:
                   trimmed_end_dates.append(np.array(dates)[mask][j])
               break
       else:
           trimmed_end_dates.append(end)

    return start_dates, trimmed_end_dates

# The following are examples of the heatwave definition with gap days options
# start_dates_new, end_dates_new = find_heatwaves(hotdays_Whs.date, hotdays_Whs['hotday_indicator'], minimum_length=3, gap_days=1, gap_day_window=8)
# start_dates_new2, end_dates_new2 = find_heatwaves(hotdays_Whs.date, hotdays_Whs['hotday_indicator'], minimum_length=5, gap_days=1, gap_day_window=5)

def find_Pezza_heatwaves(max_hotdays, min_hotdays, dates, minimum_length,
                         tolerance, gap_day_window):
    '''

    Parameters
    ----------
    max_hotdays : TYPE
        DESCRIPTION. 0/1 indicator of days that surpass max temp 90th quantile
    min_hotdays : TYPE
        DESCRIPTION. 0/1 indicator of days that surpass min temp 90th quantile
    dates : TYPE
        DESCRIPTION. vector of dates associated with max_hotdays and min_hotdays
    minimum_length : TYPE
        DESCRIPTION. minimum number of consecutive days for a heatwave to occur
    tolerance:
        DESCRIPTION. number of days per gap_day_window that can be non-active
                    and a heatwave continue. This will stay 0 for Pezza approach
    gap_day_window:
        DESCRIPTION. length of days that we replenish tolerance days after during a heatwave,
                    thiswill stay 0 for Pezza approach.

    Returns
    -------
    extended_start_dates : TYPE
        DESCRIPTION. start dates of heatwaves
    end_dates : TYPE
        DESCRIPTION. end dates of heatwaves

    '''
    # Find days where its both a max and min hotday
    active = (max_hotdays == 1) & (min_hotdays == 1)
    
    start_dates, end_dates = [], []
    in_heatwave = False
    start_time = None
    tolerance_left = tolerance
    hotday_count = 0
    total_days_in_window = 0 
    
    # Now go through the process of finding those consecutive hotdays
    # This is the same as in the normal heatwave function
    for i in range(len(active)):
        if active[i]:
            if not in_heatwave:
                # Start new heatwave
                in_heatwave = True
                start_time = dates[i]
                hotday_count = 1
                total_days_in_window = 1
                tolerance_left = tolerance
            else:
                # Continue existing heatwave
                hotday_count += 1
                total_days_in_window += 1

                # Reset tolerance if window reached
                if total_days_in_window >= gap_day_window:
                    tolerance_left = tolerance
                    total_days_in_window = 0  # start a new window
        else:
            if in_heatwave:
                total_days_in_window += 1
                if tolerance_left > 0:
                    # Allow a cool day, but don't add to hotday_count
                    tolerance_left -= 1
                else:
                    # No tolerance left → heatwave ends
                    if hotday_count >= minimum_length:
                        end_time = dates[i - 1]
                        start_dates.append(start_time)
                        end_dates.append(end_time)
                    # Reset
                    in_heatwave = False
                    hotday_count = 0
                    tolerance_left = tolerance
                    total_days_in_window = 0

    # Handle if heatwave goes till end
    if in_heatwave and hotday_count >= minimum_length:
        end_time = dates.iloc[-1]
        start_dates.append(start_time)
        end_dates.append(end_time)
        
    # Trim off any non-hotdays around the end dates
    trimmed_end_dates = []
    for start, end in zip(start_dates, end_dates):
       mask = (dates >= start) & (dates <= end)
       segment = active[mask]

       # Walk backward to find last hotday
       for j in range(len(segment) - 1, -1, -1):
           if segment.iloc[j] if hasattr(segment, "iloc") else segment[j]:
               if hasattr(dates, "iloc"):
                   trimmed_end_dates.append(dates[mask].iloc[j])
               else:
                   trimmed_end_dates.append(np.array(dates)[mask][j])
               break
       else:
           trimmed_end_dates.append(end)
           
    # If the day before a heatwave starts is a max_hotday, then add that into the heatwave
    extended_start_dates = []
    for start in start_dates:
        # Find date for previous day
        previous_day = start - pd.Timedelta(days=1)
        # Check if its a max_hotday
        if (max_hotdays[dates==previous_day].iloc[0] == 1):
            # If it is, then change the start date to the previous day
            extended_start_dates.append(previous_day)
        else:
            # Otherwise, keep the start date the same
            extended_start_dates.append(start)
            
    
    return extended_start_dates, trimmed_end_dates


def build_date_range(start_dates, end_dates,frequency):
    '''
    Name: build_date_range()
    Summary: Takes the start and end dates of heatwaves and provides all heatwave dates

    Input: start_dates ~ vector of heatwave start dates
           end_dates ~ vector of heatwave end dates

    Output: date_range ~ list of list of all dates for each heatwave
    '''
    
    date_range = []
    for start, end in zip(start_dates, end_dates):
        date_range.append(pd.date_range(start,end,freq=frequency))
    return date_range
    

def find_daily_quantiles(historical_dates, historical_temperatures, my_quantile):
    '''
    Name: find_daily_quantiles()
    Summary: Finds quantiles for a given day of the year over historical data

    Input: historical_dates ~ vecotr of dates over historical data
           historical_temperatures ~ vector of temperatures over the historical data
           my_quantile ~ percentile you want to calculate

    Output: historical_quantiles ~ dataframe with month-day column and the historical quantile
    Note: This is being used for heatwave definition, so variables are temp based but can be used in other contexts

    '''
    # Isolate month and day for each date
    month_day = historical_dates.dt.strftime('%m-%d')
    # create a dataframe of month-day and temp
    daily_max_temp = pd.DataFrame({'month_day':month_day,
                                   'max_temp':historical_temperatures})
    
    # group by month-day and calculate quantile
    quantile_temperatures = (
    daily_max_temp
    .groupby('month_day')['max_temp']
    .quantile(my_quantile)
    .reset_index()
    .rename(columns={'max_temp': 'quantile_temperature'})
    )
    
    return quantile_temperatures


def describe_heatwaves(start_dates, end_dates, timeseries_dates, timeseries_temperature,
                       historical_dates, historical_temperatures, method):
    '''
    Name: describe_heatwaves()
    Summary: Gives some indices and information about the heatwaves 

    Input: start_dates ~ vector of heatwave start dates
           end_dates ~ vector of heatwave end dates
           timeseries_dates ~ vector of dates from timeseries of temp
           timeseries_temperature ~ vector of max temperature for each day of timeseries

    Output: start_dates ~ vector of dates that mark the beginning of heatwaves
            end_dates ~ vector of dates that mark the end of heatwaves
            duration ~ length of days heatwave lasted
            magnitude ~ heatwave magnitude index defined by Marengo (2025)
    '''
    
    heatwave_df = pd.DataFrame()
    heatwave_df['start_dates'] = start_dates
    heatwave_df['end_dates'] = end_dates
    heatwave_df['duration'] = pd.Series(end_dates) - pd.Series(start_dates) + pd.Timedelta(days=1)
    
    # Now calculating the Marengo 2025 heatwave magnitude index
    # 30 year 25th and 75th percentile temperature, and maximum daily temp
    quantile_temp_25 = find_daily_quantiles(historical_dates, historical_temperatures, .25)
    quantile_temp_75 = find_daily_quantiles(historical_dates, historical_temperatures, .75)
    # Find date ranges of the heatwaves
    heatwave_dates = build_date_range(start_dates, end_dates, frequency='D')
    
    # Only including magnitude index for maximum method
    if (method == 'max'):
        # Find maximum temperature over all days in timeseries
        max_daily_temp = find_max_temperatures(timeseries_dates,timeseries_temperature)
        # For each heatwave, find the max daily temp and calculate the magnitude index
        heatwave_magnitude = []
        for heatwave in heatwave_dates:
            Td_max = max_daily_temp[max_daily_temp['date'].isin(heatwave)].max_temperature
            heatwave_month_day = heatwave.strftime('%m-%d')
            T25p = quantile_temp_25[quantile_temp_25['month_day'].isin(heatwave_month_day)].quantile_temperature
            T75p = quantile_temp_75[quantile_temp_75['month_day'].isin(heatwave_month_day)].quantile_temperature
            heatwave_magnitude.append(((Td_max.values - T25p.values) / (T75p - T25p)).sum())
    
        heatwave_df['magnitude'] = heatwave_magnitude
        
    
    return heatwave_df


# Example of the above with the consecutive hot day method, and the additional gap method
# heatwaves_Whs = describe_heatwaves(start_dates_new, end_dates_new, 
#                                    flux_data_Whs.timestamp, flux_data_Whs.TA_F,
#                                    historical_data_Whs.date, historical_data_Whs.Tmax)
# print(heatwaves_Whs)
# print(heatwaves_Whs.sort_values(by = 'magnitude'))

# heatwaves_Whs_new = describe_heatwaves(start_dates_new, end_dates_new, 
#                                    flux_data_Whs.timestamp, flux_data_Whs.TA_F,
#                                    historical_data_Whs.date, historical_data_Whs.Tmax)
# print(heatwaves_Whs_new)
# print(heatwaves_Whs_new.sort_values(by = 'magnitude'))

def get_heatwave_indicator(start_dates, end_dates, daily_dates):
    '''
    Name: get_heatwave_indicator()
    Summary: This takes the start and end dates of heatwaves and returns an vector
             with a 0/1 indicator of a heatwave day (DIFFERENT THAN HOTDAYS)

    Input: start_dates ~ vector of heatwave start dates
           end_dates ~ vector of heatwave end dates
           all_dates ~ vector of all timeseries dates

    Output: heatwave_days ~ dataframe with date column and 0/1 indicator of whether 
            a day is part of a heatwave defined by find_heatwaves()
    
    '''
    # Initialize a list with all zeroes
    heatwave_days = pd.DataFrame({'date':daily_dates,
                                  'heatwave_indicator':[0]*len(daily_dates)})
    
    # For each start and end dates, create a range of dates in between
    for start, end in zip(start_dates, end_dates):
        date_range = pd.date_range(start=start,end=end,freq='D')
        heatwave_days.loc[heatwave_days['date'].isin(date_range),'heatwave_indicator'] = 1
     
    return heatwave_days
    
# Example of the above
# heatwaves = get_heatwave_indicator(start_dates_new, end_dates_new, max_temperatures_Whs.date)

# Plotting this example
# plt.figure()
# plt.scatter(max_temperatures_Whs.date, max_temperatures_Whs.max_temperature, s = .5)
# plt.scatter(max_temperatures_Whs.date[heatwaves['heatwave_indicator']==1],max_temperatures_Whs.max_temperature[heatwaves['heatwave_indicator']==1],c="red",s=.5)
# plt.show()

def fit_heatwaves(flux_dates, flux_temperature, 
                  historical_dates, historical_temperature,
                  quantile_threshold = .9,
                  window_length = 15,
                  threshold_comparison = 'greater',
                  min_heatwave_length = 3,
                  tolerance = 1,
                  gap_days_window = 8,
                  site = "Example",
                  method = "max"
                  ):
    '''
     Name: fit_heatwaves()
     Summary: This wraps defining, summarizing, and plotting heatwaves into one.  

     Input:    method ~ "max" for maximum temperature quantile approach,
                        "EHF" for EHF approach (use daily temperature for flux temp)
            
            

     Output:   heatwaves ~ a dictionary including...
               start_dates, end_dates ~ all start and end dates of each heatwave
               summary ~ the summary returned by the describe_heatwaves() function
               indicator ~ 0/1 indicator of heatwave inclusion
               periods ~ dates and max temperature for heatwave days only
               plot ~ plot of max temperatures with heatwave days in red
    '''
    # Define heatwaves using the max approach
    if (method == 'max'):
        # Find maximum temperature for each flux day
        daily_max_temperatures = find_max_temperatures(flux_dates,flux_temperature)
        # Find moving quantile window for flux data temperature
        # Default parameters are a window of 15 days and 90th quantile
        max_temperature_quantiles = moving_window_quantile(
            dates = historical_dates,
            measure = historical_temperature,
            measure_quantile = quantile_threshold,
            window_length = window_length
            )
        # Define hotdays with the maximum temperature quantiles determined
        # prior as the threshold
        print(f"Finding max temperature hot days for site {site}.")
        hotdays = define_hotdays(
            timeseries_dates = daily_max_temperatures.date,
            timeseries_temperature = daily_max_temperatures.max_temperature,
            threshold_month_day = max_temperature_quantiles.month_day,
            threshold = max_temperature_quantiles.quantiles,
            comparison = threshold_comparison
            )
        
        # Determine the start and end dates of the heatwaves
        # Default leniency is 1 gap day per every 8 days of heatwave, with min 
        # number of 3 consecutive hotdays for a heatwave
        start_dates, end_dates = find_heatwaves(
             dates = hotdays.date, 
             hotdays = hotdays.hotday_indicator,
             minimum_length = min_heatwave_length, 
             tolerance = tolerance, 
             gap_day_window = gap_days_window
             )
        
        # Get the summary of the heatwaves
        summary = describe_heatwaves(
            start_dates = start_dates, 
            end_dates = end_dates, 
            timeseries_dates = daily_max_temperatures.date, 
            timeseries_temperature = daily_max_temperatures.max_temperature,
            historical_dates = historical_dates,
            historical_temperatures = historical_temperature,
            method = method
            )
        
        # Get the vector indicator of hot days
        indicator = get_heatwave_indicator(start_dates = start_dates, 
                                           end_dates = end_dates, 
                                           daily_dates = daily_max_temperatures.date
                                           )
        # Get the max temperatures associated with each heatwave
        periods = pd.DataFrame({"date": daily_max_temperatures.date[indicator.heatwave_indicator == 1],
                                "max_temperature": daily_max_temperatures.max_temperature[indicator.heatwave_indicator == 1]})
        # Provide a plot of a heatwave
        fig, ax = plt.subplots()
        ax.scatter(daily_max_temperatures.date,daily_max_temperatures.max_temperature, s=.5, c='lightgrey')
        ax.scatter(periods.date,periods.max_temperature,s=.5,c="red")
        ax.set_title(f"Max: {site}")
        heatwave_plot = fig
        
    # Define heatwaves using the EHF approach
    elif (method == "EHF"):
        print(f"Finding EHF hotdays for site {site}.")
        EHF_df = define_EHF_hotdays(flux_dates, 
                                    flux_temperature, 
                                    historical_dates, 
                                    historical_temperature)
        hotdays = EHF_df[['date','hotday_indicator']].reset_index(drop=True)
        
        # Determine the start and end dates of the heatwaves
        # Default leniency is 1 gap day per every 8 days of heatwave, with min 
        # number of 3 consecutive hotdays for a heatwave
        start_dates, end_dates = find_heatwaves(
             dates = hotdays.date, 
             hotdays = hotdays.hotday_indicator,
             minimum_length = min_heatwave_length, 
             tolerance = tolerance, 
             gap_day_window = gap_days_window
             )
        
        # Get the summary of the heatwaves
        summary = describe_heatwaves(
            start_dates = start_dates, 
            end_dates = end_dates, 
            timeseries_dates = flux_dates, 
            timeseries_temperature = flux_temperature,
            historical_dates = historical_dates,
            historical_temperatures = historical_temperature,
            method = method
            )
        
        # Get vector indicator of hot days
        indicator = get_heatwave_indicator(start_dates = start_dates,
                                           end_dates = end_dates,
                                           daily_dates = flux_dates)
        
        # Get the max temperatures associated with each heatwave
        periods = pd.DataFrame({"date": flux_dates.loc[29:][indicator.heatwave_indicator==1],
                                "temperature": flux_temperature.loc[29:][indicator.heatwave_indicator==1]})
        
        # Provide a plot of a heatwave
        fig, ax = plt.subplots()
        ax.scatter(flux_dates,flux_temperature, s=.5, c='lightgrey')
        ax.scatter(periods.date,periods.temperature,s=.5,c="red")
        ax.set_title(f"EHF: {site}")
        heatwave_plot = fig
        
    # Create a dictionary and store all of these inside it!
    heatwaves = {
        "start_dates":start_dates,
        "end_dates":end_dates,
        "summary":summary,
        "indicator":indicator,
        "periods":periods,
        "plot":heatwave_plot
        }
    
    return heatwaves

def fit_Pezza_heatwaves(flux_dates, flux_temperature, 
                  historical_dates, historical_temperature_max,
                  historical_temperature_min,
                  min_heatwave_length = 3,
                  site = "Example"
                  ):
    # Define heatwaves using the Pezza approach
    print(f"Finding Pezza hotdays for site {site}.")
    # Find maximum and minimum temperatures for each day 
    daily_max_temperatures = find_max_temperatures(flux_dates,flux_temperature)
    daily_min_temperatures = find_min_temperatures(flux_dates,flux_temperature)
    
    # Find the 90th quantile of the moving average max and minimum historical temperatures
    max_temperature_quantiles = moving_window_quantile(
        dates = historical_dates,
        measure = historical_temperature_max,
        measure_quantile = .9,
        window_length = 31
        )
    
    min_temperature_quantiles = moving_window_quantile(
        dates = historical_dates,
        measure = historical_temperature_min,
        measure_quantile = .9,
        window_length = 31
        )
        
    # Determine hotdays by minimum and maximum temperature
    max_hotday = define_hotdays(
        timeseries_dates = daily_max_temperatures.date,
        timeseries_temperature = daily_max_temperatures.max_temperature,
        threshold_month_day = max_temperature_quantiles.month_day,
        threshold = max_temperature_quantiles.quantiles,
        comparison = "greater"
        )
        
    min_hotday = define_hotdays(
        timeseries_dates = daily_min_temperatures.date,
        timeseries_temperature = daily_min_temperatures.min_temperature,
        threshold_month_day = min_temperature_quantiles.month_day,
        threshold = min_temperature_quantiles.quantiles,
        comparison = "greater"
        )
        
    # Determine heatwaves using function for Pezza heatwaves specifically
    start_dates, end_dates = find_Pezza_heatwaves(
        dates = max_hotday.date, 
        max_hotdays = max_hotday.hotday_indicator,
        min_hotdays = min_hotday.hotday_indicator,
        minimum_length = 3, 
        tolerance = 0, 
        gap_day_window = 0
        )
        
    # Get the summary of the heatwaves
    summary = describe_heatwaves(
        start_dates = start_dates, 
        end_dates = end_dates, 
        timeseries_dates = daily_max_temperatures.date, 
        timeseries_temperature = daily_max_temperatures.max_temperature,
        historical_dates = historical_dates,
        historical_temperatures = historical_temperature_max,
        method = "greater"
        )
        
    # Get the vector indicator of hot days
    indicator = get_heatwave_indicator(start_dates = start_dates, 
                                       end_dates = end_dates, 
                                       daily_dates = daily_max_temperatures.date
                                       )
    # Get the max temperatures associated with each heatwave
    periods = pd.DataFrame({"date": daily_max_temperatures.date[indicator.heatwave_indicator == 1],
                            "max_temperature": daily_max_temperatures.max_temperature[indicator.heatwave_indicator == 1],
                            "min_temperature": daily_min_temperatures.min_temperature[indicator.heatwave_indicator==1]})
    # Provide a plot of a heatwave
    fig, ax = plt.subplots()
    ax.scatter(daily_max_temperatures.date,daily_max_temperatures.max_temperature, s=.5, c='lightgrey')
    ax.scatter(periods.date,periods.max_temperature,s=.5,c="red")
    ax.set_title(f"Pezza: {site}")
    heatwave_plot = fig
  
    # Create a dictionary and store all of these inside it!
    heatwaves = {
        "start_dates":start_dates,
        "end_dates":end_dates,
        "summary":summary,
        "indicator":indicator,
        "periods":periods,
        "plot":heatwave_plot
        }

    return heatwaves

def calculate_moisture(timeseries_dates,timeseries_moisture,start_dates,end_dates):
    '''
    Name: calculate_moisture()
    Summary: This provides the average moisture conditions during a given heatwave
    evet.

    Input: timeseries_dates ~ daily dates over the moisture timeseries
           timeserues_moisture ~ daily moisture conditions of interest over the timeseries
           start_dates ~ vector of heatwave start dates
           end_dates ~ vector of heatwave end dates
           

    Output: moisture_averages ~ dataframe of start date, end date, and average
            moisture conditions over that time period
    '''
    moisture_averages = []
    moisture_totals = []
    # Loop through each heatwave
    for start, end in zip(start_dates, end_dates):
        # Create date range for the heatwave
        date_range = pd.date_range(start=start,end=end,freq='D')
        # Find moisture conditions during that period
        moisture = timeseries_moisture[timeseries_dates.isin(date_range)]
        # If we have moisture for that dataset
        if len(moisture) > 0:
            # Calculate average moisture
            average = sum(moisture) / len(moisture)
            total = sum(moisture)
            # Add to list of heatwave moisture averages
        else:
            average = pd.NA
            total = pd.NA
        moisture_averages.append(average)
        moisture_totals.append(total)
        
    
    heatwave_moisture = pd.DataFrame({'start_date':list(start_dates),
                                      'end_date':list(end_dates),
                                      'moisture_average':list(moisture_averages),
                                      'moisture_total':list(moisture_totals)
                                      })
    
    return heatwave_moisture


print("Heatwave defining functions all loaded.")