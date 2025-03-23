from datetime import datetime
import numpy as np
from utilities import data_class
"""
Check the table flights for missing values and think of ways to resolve them.
"""

'''
The zero values in the departure and arrival rows will be removed from the dataset. 
This is because the rows with missing data are not considered as zero records; 
meaning that if a flight has no delay, it is simply recorded with the number zero, and not as a missing value. 
Therefore, we assume that the rows with null values can be removed for this reason.
'''

def remove_nan_values(flights_df):
    return flights_df[(~ flights_df['dep_time'].isna()) & (~ flights_df['arr_time'].isna())]

"""
Look for duplicates in the flight table. Take into account that here a flight
number can occur multiple times, only count it as duplicate when the same flight
appears multiple times.
"""

def remove_duplicates(flights_df):
    unique_columns = ['year', 'month', 'day', 'flight', 'dep_time', 'arr_time', 'origin', 'dest']
    unique_df = flights_df.drop_duplicates(subset=unique_columns, keep='first')
    return unique_df
    
"""
Convert the (schedueled and actual) arrival departure and departure moments
to datetime objects.
"""
def convert_time_columns(flights_df):
    convert_cols = ['dep_time', 'sched_dep_time', 'arr_time', 'sched_arr_time']
    processed_df = flights_df.copy()

    for col in convert_cols:
        new_col = col.replace('time', 'date')
        bool_mask = processed_df[col].notna()
        
        # Create a temporary Series for the new column
        temp_series = pd.Series(np.nan, index=processed_df.index, dtype='object')
        
        # Fill the temporary Series with formatted time strings where applicable
        temp_series[bool_mask] = (
            processed_df.loc[bool_mask, col]
            .astype(int)
            .astype(str)
            .str.zfill(4)
        )
        
        # Convert to datetime
        temp_series = pd.to_datetime(
            processed_df['year'].astype(str) + '-' +
            processed_df['month'].astype(str) + '-' +
            processed_df['day'].astype(str) + ' ' +
            temp_series.str[:2] + ':' +
            temp_series.str[2:],
            format='%Y-%m-%d %H:%M', errors='coerce'
        )
        
        # Assign the temporary Series back to the DataFrame
        processed_df[new_col] = temp_series

    return processed_df

def adjust_flight_dates(flights_df):
    processed_df = flights_df.copy()

    # Correction for time travel: increase arr_date and sched_arr_date by 1 day if dep_date > arr_date
    time_travel_mask = processed_df['dep_date'] > processed_df['arr_date']
    processed_df.loc[time_travel_mask, ['arr_date', 'sched_arr_date']] += pd.Timedelta(days=1)

    # Mask for flights before January 1, 2023 at 05:00
    date_mask = processed_df['dep_date'] < datetime(year=2023, month=1, day=1, hour=5, minute=0)

    # Correction for scheduled departure dates: decrease sched_dep_date by 1 day if dep_date < sched_dep_date
    sched_dep_correction_mask = (date_mask) & (processed_df['dep_date'] < processed_df['sched_dep_date'])
    processed_df.loc[sched_dep_correction_mask, 'sched_dep_date'] -= pd.Timedelta(days=1)

    # Correction for scheduled arrival dates: decrease sched_arr_date by 1 day if arr_date < sched_arr_date
    sched_arr_correction_mask = (date_mask) & (processed_df['arr_date'] < processed_df['sched_arr_date'])
    processed_df.loc[sched_arr_correction_mask, 'sched_arr_date'] -= pd.Timedelta(days=1)

    return processed_df

"""
Write a function that checks whether the data in flights is in order. That
is, verify that the air time , dep time , sched dep time etc. match for each
flight. If not, think of ways to resolve it if this is not the case.
"""

def calculate_delays(flights_df):
    processed_df = flights_df.copy()

    # Calculate departure delay in minutes
    processed_df['dep_date_delay'] = (processed_df['dep_date'] - processed_df['sched_dep_date']) / pd.Timedelta(minutes=1)

    # Calculate arrival delay in minutes
    processed_df['arr_date_delay'] = (processed_df['arr_date'] - processed_df['sched_arr_date']) / pd.Timedelta(minutes=1)

    # Calculate total delay as the sum of departure and arrival delays
    processed_df['total_delay'] = processed_df['dep_date_delay'] + processed_df['arr_date_delay']

    return processed_df
# A second check if the arr_date and dep_date need to be shifted an extra day. Because there are currently delays with e.g. -1100 minutes, i.e. this could mean that the arr_date and dep_date need to be shifted one day extra

def adjust_negative_delays(flights_df, threshold=-600):
    processed_df = flights_df.copy()

    # Identify rows where delays are below the threshold
    dep_delay_mask2 = processed_df['dep_date_delay'] <= threshold
    arr_delay_mask2 = processed_df['arr_date_delay'] <= threshold

    # Adjust dates for identified rows
    processed_df.loc[dep_delay_mask2, 'dep_date'] += pd.Timedelta(days=1)
    processed_df.loc[arr_delay_mask2, 'arr_date'] += pd.Timedelta(days=1)

    # Recalculate delays after adjustments
    processed_df = calculate_delays(processed_df)

    return processed_df

'''
Because there is almost no equality in the given air_time and our calculated air_time, we only compare if the data in flights is in order for the delays.
'''

def check_delay_equality(flights_df):
    dep_delay_mask = flights_df['dep_delay'] == flights_df['dep_date_delay']
    arr_delay_mask = flights_df['arr_delay'] == flights_df['arr_date_delay']
    flights_df.loc[:, 'delay_eq'] = False
    flights_df.loc[(dep_delay_mask) & (arr_delay_mask), 'delay_eq'] = True

"""
Create a column that contains the local arrival time, incorporating the time
difference between arrival and departure airport.
"""
def merge_timezone_info(flights_df, airports_df):
    dest_df = airports_df[['faa', 'tz']].rename(columns={'faa': 'dest', 'tz': 'dest_tz'})
    flights_df = flights_df.merge(dest_df, on='dest', how='left')
    
    timezones = {
        'BQN': -4,
        'SJU': -4,
        'STT': -4,
        'PSE': -4
    }

    flights_df['dest_tz'] = flights_df['dest_tz'].fillna(
        flights_df['dest'].map(timezones)
    )
    
    return flights_df

def convert_arr_date_to_gmt5(flights_df):
    def tz_to_tz_string(tz):
        if pd.isna(tz):
            return None
        if tz < 0:
            return f'Etc/GMT+{abs(int(tz))}'
        else:
            return f'Etc/GMT-{int(tz)}'

    for tz, indices in flights_df.groupby('dest_tz').groups.items():
        if pd.isna(tz):
            continue
        localized_date = flights_df.loc[indices, 'arr_date'].dt.tz_localize(tz_to_tz_string(tz))
        flights_df.loc[indices, 'arr_date_gmt5'] = localized_date.dt.tz_convert('Etc/GMT+5')

    flights_df['arr_date_gmt5'] = flights_df['arr_date_gmt5'].dt.tz_localize(None)

'''
Because the air_time that is given in the dataset is sparse when equal to our calculated air_time, which is the difference between dep_date and arr_date_gmt5. 
We made the assumption that our calculated air_time would be so called block_time. Which could be seen as the time difference when the brake blocks are on (arr_date_gmt5) and brake blocks are off (dep_date).
So that could be seen as gross air_time. The positive difference between the given air_time from the dataset and our calculated block_time is the taxi_time for which an airplane takes. 
We will use taxi_time as a statistic in our dashboard.
'''

def calculate_block_and_taxi_time(flights_df):
    flights_df['block_time'] = (flights_df['arr_date_gmt5'] - flights_df['dep_date']) / pd.Timedelta(minutes=1)
    flights_df['taxi_time'] = np.nan
    block_mask = flights_df['block_time'] > flights_df['air_time']
    flights_df.loc[block_mask, 'taxi_time'] = flights_df['block_time'] - flights_df['air_time']

"""
In addition, information on the different types of planes and airlines will be
important. Consider studying what the effect of the wind or precipitation is on
different plane types.
"""

import pandas as pd
import matplotlib.pyplot as plt

def analyze_weather_effect_by_plane_model(top_n_models, sample_size):
    query = """
    SELECT 
      f.dep_delay,
      f.tailnum,
      f.origin,
      f.year,
      f.month,
      f.day,
      f.hour,
      f.minute,
      CASE WHEN f.minute >= 30 THEN 
            CASE WHEN f.hour = 23 THEN 0 ELSE f.hour + 1 END 
           ELSE f.hour END AS dep_hour_adj,
      p.model AS plane_model,
      w.wind_speed,
      w.precip
    FROM flights f
    JOIN planes p ON f.tailnum = p.tailnum
    JOIN weather w ON f.origin = w.origin 
       AND f.year = w.year 
       AND f.month = w.month 
       AND f.day = w.day 
       AND (CASE WHEN f.minute >= 30 THEN 
                  CASE WHEN f.hour = 23 THEN 0 ELSE f.hour + 1 END 
                 ELSE f.hour END) = w.hour
    WHERE f.dep_delay IS NOT NULL
      AND w.wind_speed IS NOT NULL
    LIMIT ?
    """
    conn = data_class.conn
    df = pd.read_sql_query(query, conn, params=(sample_size,))
    
    # Ensure precipitation is numeric; fill missing with 0.
    df['precip'] = pd.to_numeric(df['precip'], errors='coerce').fillna(0)
    
    # Count flights per model and select top N models.
    model_counts = df['plane_model'].value_counts().head(top_n_models)
    top_models = model_counts.index.tolist()
    print("Top plane models by number of flights:", top_models)
    
    # Create scatter plots for each of the top models.
    for model in top_models:
        sub_df = df[df['plane_model'] == model]
        if sub_df.empty:
            continue
        plt.figure(figsize=(14, 6))
        
        # Plot wind speed vs. departure delay.
        plt.subplot(1, 2, 1)
        plt.scatter(sub_df['wind_speed'], sub_df['dep_delay'], alpha=0.5)
        plt.xlabel("Wind Speed")
        plt.ylabel("Departure Delay (min)")
        plt.title(f"Model: {model} | Wind Speed vs. Dep Delay")
        
        # Plot precipitation vs. departure delay.
        plt.subplot(1, 2, 2)
        plt.scatter(sub_df['precip'], sub_df['dep_delay'], alpha=0.5, color='orange')
        plt.xlabel("Precipitation (mm/hr)")
        plt.ylabel("Departure Delay (min)")
        plt.title(f"Model: {model} | Precipitation vs. Dep Delay")
        
        plt.tight_layout()
        plt.show()
    
    return df