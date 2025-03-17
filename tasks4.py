import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

# importing the flights data from the database
conn = sqlite3.Connection('Data/flights_database.db')
cursor = conn.cursor()
query_flights = f'SELECT * FROM flights'
cursor.execute(query_flights)
data_rows = cursor.fetchall()
flights_df = pd.DataFrame(data_rows, columns=[col[0] for col in cursor.description])

"""
Check the table flights for missing values and think of ways to resolve them.
"""

"""
Look for duplicates in the flight table. Take into account that here a flight
number can occur multiple times, only count it as duplicate when the same flight
appears multiple times.
"""

"""
Convert the (schedueled and actual) arrival departure and departure moments
to datetime objects.
"""
convert_cols = ['dep_time', 'sched_dep_time', 'arr_time', 'sched_arr_time']


def convert_time(col):
    new_col = col.replace('time', 'date')
    bool_mask = flights_df[col].notna()
    flights_df[new_col] = np.nan
    flights_df.loc[bool_mask, new_col] = flights_df.loc[bool_mask, col].astype(int).astype(str).str.zfill(4)
    flights_df[new_col] = pd.to_datetime(flights_df['year'].astype(str) + '-' +
                                         flights_df['month'].astype(str) + '-' +
                                         flights_df['day'].astype(str) + ' ' +
                                         flights_df[new_col].str[:2] + ':' +
                                         flights_df[new_col].str[2:],
                                         format='%Y-%m-%d %H:%M', errors='coerce')


for col in convert_cols:
    convert_time(col)

# Shifting the arr_date and sched_arr_date one day up, when dep_date is greater than arr_date
# because it is not yet invented: time travelling
flights_df.loc[flights_df['dep_date'] > flights_df['arr_date'], ['arr_date', 'sched_arr_date']] = flights_df.loc[
                                                                                                      flights_df[
                                                                                                          'dep_date'] >
                                                                                                      flights_df[
                                                                                                          'arr_date'], [
                                                                                                          'arr_date',
                                                                                                          'sched_arr_date']] + pd.Timedelta(
    days=1)

date_mask = flights_df['dep_date'] < datetime(year=2023, month=1, day=1, hour=5, minute=00)

# Departure dates
# The same principle hold for the flights where the dataset starts, e.g. sched_dep_time is 2023-01-01 20:38, it is probably not departing on 2023-01-01 00:01, so we shift in those cases a day down
flights_df.loc[(date_mask) & (flights_df['dep_date'] < flights_df['sched_dep_date']), 'sched_dep_date'] = \
flights_df.loc[(date_mask) & (flights_df['dep_date'] < flights_df['sched_dep_date']), 'sched_dep_date'] - pd.Timedelta(
    days=1)

# Arrival dates as well
flights_df.loc[(date_mask) & (flights_df['arr_date'] < flights_df['sched_arr_date']), 'sched_arr_date'] = \
flights_df.loc[(date_mask) & (flights_df['arr_date'] < flights_df['sched_arr_date']), 'sched_arr_date'] - pd.Timedelta(
    days=1)

"""
Write a function that checks whether the data in flights is in order. That
is, verify that the air time , dep time , sched dep time etc. match for each
flight. If not, think of ways to resolve it if this is not the case.
"""

flights_df['dep_date_delay'] = (flights_df['dep_date'] - flights_df['sched_dep_date']) / pd.Timedelta(minutes=1)
flights_df['arr_date_delay'] = (flights_df['arr_date'] - flights_df['sched_arr_date']) / pd.Timedelta(minutes=1)
# Still not sure how to calulcate air_time
# I thought it would be the difference between arrival at destination and departure at origin
flights_df['air_time_date'] = (flights_df['arr_date'] - flights_df['dep_date']) / pd.Timedelta(minutes=1)

"""
Create a column that contains the local arrival time, incorporating the time
difference between arrival and departure airport.
"""

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

if __name__ == "__main__":
    try:
        df_joined = analyze_weather_effect_by_plane_model(top_n_models=10, sample_size=10000)
    except Exception as e:
        print("Error during analysis", e)