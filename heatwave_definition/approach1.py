# This script will explore the different approaches for the definition of heatwaves

# Importing packages
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from matplotlib import pyplot as plt

#  The following is all data prep for the examples in this page

# Setting all columns to be printed
# pd.set_option('display.max_columns', None)

# Import data: starting with US-Whs site
# flux_data_Whs = loadAMFFile('/Users/marleeyork/Documents/project2/AMFdata/AMF_US-Whs_FLUXNET_SUBSET_HH_2007-2020_3-5.csv', 
#                             measures = ['TIMESTAMP_START','TA_F','SW_IN_F','VPD_F','P_F','NEE_VUT_REF','RECO_NT_VUT_REF','GPP_NT_VUT_REF'])
# historical_data = pd.read_csv('/Users/marleeyork/Documents/project2/extracted_daily_climate_data_wide_tmax.csv')
# historical_data_Whs = historical_data.loc[:,['date','US-Whs']]
# historical_data_Whs = historical_data_Whs.rename(columns={"date":"date", "US-Whs":"Tmax"})
# flux_data_Whs['timestamp'] = pd.to_datetime(flux_data_Whs['TIMESTAMP_START'], format='%Y%m%d%H%M')
# historical_data_Whs['date'] = pd.to_datetime(historical_data_Whs['date'], format='%Y-%m-%d')


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


# Example of the find_max_temperatures() and find_min_temperatures()
# max_temperatures_Whs = find_max_temperatures(flux_data_Whs.timestamp, flux_data_Whs.TA_F)
# min_temperatures_Whs = find_min_temperatures(flux_data_Whs.timestamp, flux_data_Whs.TA_F)

# plt.figure()
# plt.scatter(max_temperatures_Whs.date, max_temperatures_Whs.max_temperature, color= 'red', s = .5)
# plt.scatter(min_temperatures_Whs.date, min_temperatures_Whs.min_temperature, color = 'blue', s = .5)
# plt.show()

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


# An example of the moving_window_quantiles() function, taking in US-Whs historical dates 
# and daily maximum temperature and calculating the 90th percentile within a 15 day window.
# max_temperature_90th_quantiles = moving_window_quantile(
#     dates = historical_data_Whs.date, 
#     measure = historical_data_Whs.Tmax, 
#     measure_quantile = .9,
#     window_length = 15
#     )

# plt.scatter(max_temperature_90th_quantiles['month_day'], max_temperature_90th_quantiles['quantiles'])
# plt.show()    


def define_hotdays(timeseries_dates, timeseries_temperature, threshold_month_day, threshold, comparison = "greater"):
    '''
    Name: define_hotdays()
    Summary: Returns an indicator (0/1) whether each day is a hot day or not based on
             whether is greater than, less than, or equal to some daily threshold (e.g.,
             90th quantile of maximum temperature or EHF value)    

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

# Example of the define_hotdays() function, taking in historical flux Tmax temperatures
# (found using find_max_temperatures()) and comparing them to a threshold of
# the historical 90th quantile of Tmax over a 15 day window (using moving_window_quantiles())
# max_temperatures_Whs = find_max_temperatures(flux_data_Whs.timestamp, flux_data_Whs.TA_F)
# max_temperature_90th_quantiles = moving_window_quantile(
#    dates = historical_data_Whs.date, 
#     measure = historical_data_Whs.Tmax, 
#     measure_quantile = .9,
#    window_length = 15
#     )
# hotdays_Whs = define_hotdays(
#     timeseries_dates = max_temperatures_Whs['date'],
#     timeseries_temperature = max_temperatures_Whs['max_temperature'],
#     threshold_month_day = max_temperature_90th_quantiles['month_day'],
#     threshold = max_temperature_90th_quantiles['quantiles'],
#     comparison = "greater"
#     )
# print(hotdays_Whs)

# Plotting the above example
# hotdays = max_temperatures_Whs[hotdays_Whs.hotday_indicator==1]
# plt.scatter(max_temperatures_Whs.date, max_temperatures_Whs.max_temperature, s = .5)
# plt.scatter(hotdays.date, hotdays.max_temperature, color="red", s=.5)
# plt.show()

def define_EHF_hotdays(timeseries_dates, timeseries_temperature, 
                       historical_dates, historical_temperature):
    '''
    Name: define_EHF_hotdays()
    Summary: Returns an indicator (0/1) whether each day is a hot day or not based on
             the Excess Heat Factor described by Perkins & Alexander (2013)   

    Input: timeseries_dates ~ dates associated with the timeseries we want to define as hot or not
           timeseries_temperature ~ temperatures associated with each day of the timeseries
           historical_dates ~ vector of dates over historical/climatological data
           historical_average_temperature ~ this must be DAILY AVERAGES, not min/max

    Output: EHF_hotdays ~ indicator vector (0 or 1) of length timeseries that defines
                       each day is 1 = hot day, or 0 = not hot day based on EHF
    '''
    
    # need to fill this in whenever I get daily averages
    
    return EHF_hotdays



def find_consecutive_hotdays(dates, hotdays, minimum_length):
    '''
    Name: find_consecutive_hotdays()
    Summary: This finds periods of time with consecutive hot days for heatwave definition     

    Input: dates ~ dates across the timeseries of interest
           hotdays ~ binary (0/1) indicator of hot day, received from define_hotdays()
           minimum_length ~ minimum number of contiguous hotdays to be considered heatwave

    Output: start_dates ~ vector of dates that mark the beginning of heatwaves
            end_dates ~ vector of dates that mark the end of heatwaves
    '''
    # Loop through hotdays
    start_indices = []
    end_indices = []
    i = 0
    # Loop through each day
    while i < len(hotdays) - minimum_length + 1:
        # If the previous day was a hotday, move forward to the next
        if (i > 0):
            if (hotdays[i-1] == 1):
                i += 1
                continue
        
        days_forward = 0
 

        # Assuming heatwave, set to True
        while True:
            # Iteratively move forward a day and check if its a hotday
            # If it is a hot day...
            if (hotdays[i+days_forward] == 1):
                print("hotday" + str(days_forward))
                # and its the minimum length hot day in a row, add i as
                # the start index (heatwave has started)
                if (days_forward == minimum_length):
                    print("we have a heatwave! starting" + str(dates[i]))
                    start_indices.append(i)
                    # If we are in a heatwave, but tomorrow is not a hotday, 
                    # add the current day as the end index, set i to the next day, and leave the loop
                if ((days_forward >= minimum_length) & (hotdays[i+days_forward+1] == 0)):
                    print("heatwave ends at " + str(dates[i+days_forward]))
                    end_indices.append(i+days_forward)
                    break
                # Move onto next day to see if heatwave continues
                days_forward += 1
            
            # If its not a hot day, break and move to next day
            else: break
        # If its not a hot day, or the heatwave is over, 
        i += 1
    
    start_dates = dates[start_indices]
    end_dates = dates[end_indices]
    
    return start_dates, end_dates

                    
# This is an example of the find_consecutive_hotdays() function
# start_dates, end_dates = find_consecutive_hotdays(dates = hotdays_Whs.date, 
#                                                   hotdays = hotdays_Whs.hotday_indicator, 
#                                                   minimum_length = 3
#                                                   )                
# print([start_dates,end_dates])

def find_heatwaves(dates, hotdays, minimum_length, gap_days, gap_day_window):
    '''
    Name: find_heatwaves()
    Summary: This finds periods of time with consecutive hot days for heatwave definition ,
           with the opportunity to add gap days    

    Input: dates ~ dates across the timeseries of interest
           hotdays ~ binary (0/1) indicator of hot day, received from define_hotdays()
           minimum_length ~ minimum number of contiguous hotdays to be considered heatwave
           gap_days ~ number of gap days allowed per gap_day_window
           gap_day_window ~ number of days that gap_days can fall in 

    Output: start_dates ~ vector of dates that mark the beginning of heatwaves
            end_dates ~ vector of dates that mark the end of heatwaves
    '''
    
    # Loop through hotdays
    start_indices = []
    end_indices = []
    i = 0
    # Loop through each day
    while i < len(hotdays) - minimum_length + 1:
        # If the previous day was a hotday, move forward to the next
        if (i > 0):
            if (hotdays[i-1] == 1):
                i += 1
                continue
        
        days_forward = 0
        gaps_left = gap_days
        # Assuming heatwave, set to True
        while True:
            # print("Gaps left:" + str(gaps_left))
            # Iteratively move forward a day and check if its a hotday
            # If it is a hot day...
            
            # If its a hot day
            if (hotdays[i+days_forward] == 1):
                # print("hotday" + str(days_forward+1))
                # if its the third consecutive hot day, start the heatwave
                if (days_forward+1 == minimum_length):
                    # print("we have a heatwave! starting" + str(dates[i]))
                    start_indices.append(i)

                # If we are already in a heatwave, tomorrow is not a heatwave, 
                # and we have no more gap days, make today the end of the heatwave
                if ((days_forward+1 >= minimum_length) & (hotdays[i+days_forward+1] == 0) & (gaps_left==0)):
                    print("heatwave ends at " + str(dates[i+days_forward]))
                    end_indices.append(i+days_forward)
                    break
                # Move onto next day to see if heatwave continues
                days_forward += 1
                
            # If we are in a heatwave, but its not a hot day, but we have a gap day available
            # then subtract a gap day and move forward
            elif ((gaps_left > 0) & (days_forward+1 > minimum_length)):
                gaps_left -= 1
                days_forward += 1
                ("We used a gap day! " + str(gaps_left) + " left!")
            
            # If we are in a heatwave, but its not a hot day, and we have no more gaps left,
            # then make yesterday the 
            elif ((gaps_left == 0) & ((days_forward+1) > minimum_length)):
                # print("heatwave ending!")
                end_indices.append(i+days_forward-2)
                break
            
            # Otherwise, its not a heatwave and we just need to leave
            else: break
        
            # if we make it through a gap day window length, add another gap day
            if ((days_forward) % gap_day_window == 0):
                gaps_left += gap_days
            
        # Move to the next day to check if it starts a heatwave
        i += 1
    
    # If the beginning or end dates are not hotdays, cut them off of the heatwave
    for j in range(len(start_indices)):
        if (hotdays.iloc[start_indices[j]] == 0):
            start_indices[j] += 1
        if (hotdays.iloc[end_indices[j]] == 0):
            end_indices[j] -= 1
    
    start_dates = dates[start_indices]
    end_dates = dates[end_indices]
    
   
    return start_dates, end_dates

# The following are examples of the heatwave definition with gap days options
# start_dates_new, end_dates_new = find_heatwaves(hotdays_Whs.date, hotdays_Whs['hotday_indicator'], minimum_length=3, gap_days=1, gap_day_window=8)
# start_dates_new2, end_dates_new2 = find_heatwaves(hotdays_Whs.date, hotdays_Whs['hotday_indicator'], minimum_length=5, gap_days=1, gap_day_window=5)

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
                       historical_dates, historical_temperatures):
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
    heatwave_df['start_dates'] = start_dates.reset_index(drop=True)
    heatwave_df['end_dates'] = end_dates.reset_index(drop=True)
    heatwave_df['duration'] = end_dates.reset_index(drop=True) - start_dates.reset_index(drop=True) + pd.Timedelta(days=1)
    
    # Now calculating the Marengo 2025 heatwave magnitude index
    # 30 year 25th and 75th percentile temperature, and maximum daily temp
    quantile_temp_25 = find_daily_quantiles(historical_dates, historical_temperatures, .25)
    quantile_temp_75 = find_daily_quantiles(historical_dates, historical_temperatures, .75)
    # Find date ranges of the heatwaves
    heatwave_dates = build_date_range(start_dates, end_dates, frequency='D')
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
        heatwave_days[heatwave_days['date'].isin(date_range)].heatwave_indicator = 1
    
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
                  gap_days = 1,
                  gap_days_window = 8,
                  site = "Example"
                  ):
    '''
    # Name: fit_heatwaves()
    # Summary: This wraps defining, summarizing, and plotting heatwaves into one.  

    # Input:
    #        
    #        

    # Output:   heatwaves ~ a dictionary including...
    #           start_dates, end_dates ~ all start and end dates of each heatwave
    #           summary ~ the summary returned by the describe_heatwaves() function
    #           indicator ~ 0/1 indicator of heatwave inclusion
    #           periods ~ dates and max temperature for heatwave days only
    #           plot ~ plot of max temperatures with heatwave days in red
    '''
    
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
         gap_days = gap_days, 
         gap_day_window = gap_days_window
         )
    # Get the summary of the heatwaves
    summary = describe_heatwaves(
         start_dates = start_dates, 
         end_dates = end_dates, 
         timeseries_dates = daily_max_temperatures.date, 
         timeseries_temperature = daily_max_temperatures.max_temperature,
         historical_dates = historical_dates,
         historical_temperatures = historical_temperature
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
    ax.scatter(daily_max_temperatures.date,daily_max_temperatures.max_temperature, s=.5)
    ax.scatter(periods.date,periods.max_temperature,s=.5,c="red")
    ax.set_title(site)
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

# heatwaves_Whs = fit_heatwaves(
#     flux_dates = flux_data_Whs.timestamp, 
#     flux_temperature = flux_data_Whs.TA_F, 
#     historical_dates = historical_data_Whs.date,
#     historical_temperature = historical_data_Whs.Tmax,
#     quantile_threshold = .9,
#     window_length = 15,
#     threshold_comparison = 'greater',
#     min_heatwave_length = 3,
#     gap_days = 1,
#     gap_days_window = 8,
#    site = "US-Whs"
#     )

# print(heatwaves_Whs["start_dates"])
# print(heatwaves_Whs["summary"])
# plt.show(heatwaves_Whs["plot"])
print("Heatwave defining functions all loaded.")


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
        # Calculate average moisture
        average = sum(moisture) / len(moisture)
        total = sum(moisture)
        # Add to list of heatwave moisture averages
        moisture_averages.append(average)
        moisture_totals.append(total)
        
    
    heatwave_moisture = pd.DataFrame({'start_date':list(start_dates),
                                      'end_date':list(end_dates),
                                      'moisture_average':list(moisture_averages),
                                      'moisture_total':list(moisture_totals)
                                      })
    
    return heatwave_moisture