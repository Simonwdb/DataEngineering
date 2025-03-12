from utilities import *

data_class = Data()

# importing the flights data from the database
conn = sqlite3.Connection('Data/flights_database.db')
cursor = conn.cursor()
query_flights = f'SELECT * FROM flights'
data_class.cursor.execute(query_flights)
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
def convert_time_columns(flights_df):
    convert_cols = ['dep_time', 'sched_dep_time', 'arr_time', 'sched_arr_time']
    
    def convert_time(col):
        new_col = col.replace('time', 'date')
        bool_mask = flights_df[col].notna()
        
        flights_df[new_col] = np.nan
        flights_df[new_col] = flights_df[new_col].astype(object)
        
        flights_df.loc[bool_mask, new_col] = flights_df.loc[bool_mask, col].astype(int).astype(str).str.zfill(4)
        
        flights_df[new_col] = pd.to_datetime(
            flights_df['year'].astype(str) + '-' +
            flights_df['month'].astype(str) + '-' +
            flights_df['day'].astype(str) + ' ' +
            flights_df[new_col].str[:2] + ':' +
            flights_df[new_col].str[2:],
            format='%Y-%m-%d %H:%M', errors='coerce'
        )
    
    for col in convert_cols:
        convert_time(col)

def adjust_flight_dates(flights_df):
    # Correction for time travel: increase arr_date and sched_arr_date by 1 day if dep_date > arr_date
    time_travel_mask = flights_df['dep_date'] > flights_df['arr_date']
    flights_df.loc[time_travel_mask, ['arr_date', 'sched_arr_date']] += pd.Timedelta(days=1)

    # Mask for flights before January 1, 2023 at 05:00
    date_mask = flights_df['dep_date'] < datetime(year=2023, month=1, day=1, hour=5, minute=0)

    # Correction for scheduled departure dates: decrease sched_dep_date by 1 day if dep_date < sched_dep_date
    flights_df.loc[(date_mask) & (flights_df['dep_date'] < flights_df['sched_dep_date']), 'sched_dep_date'] -= pd.Timedelta(days=1)

    # Correction for scheduled arrival dates: decrease sched_arr_date by 1 day if arr_date < sched_arr_date
    flights_df.loc[(date_mask) & (flights_df['arr_date'] < flights_df['sched_arr_date']), 'sched_arr_date'] -= pd.Timedelta(days=1)

"""
Write a function that checks whether the data in flights is in order. That
is, verify that the air time , dep time , sched dep time etc. match for each
flight. If not, think of ways to resolve it if this is not the case.
"""

def calculate_delays(flights_df):
    flights_df['dep_date_delay'] = (flights_df['dep_date'] - flights_df['sched_dep_date']) / pd.Timedelta(minutes=1)
    flights_df['arr_date_delay'] = (flights_df['arr_date'] - flights_df['sched_arr_date']) / pd.Timedelta(minutes=1)

# A second check if the arr_date and dep_date need to be shifted an extra day. Because there are currently delays with e.g. -1100 minutes, i.e. this could mean that the arr_date and dep_date need to be shifted one day extra

def adjust_negative_delays(flights_df, threshold=-600):
    dep_delay_mask2 = flights_df['dep_date_delay'] <= threshold
    arr_delay_mask2 = flights_df['arr_date_delay'] <= threshold

    flights_df.loc[dep_delay_mask2, 'dep_date'] += pd.Timedelta(days=1)
    flights_df.loc[arr_delay_mask2, 'arr_date'] += pd.Timedelta(days=1)
    calculate_delays(flights_df)


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

"""
In addition, information on the different types of planes and airlines will be
important. Consider studying what the effect of the wind or precipitation is on
different plane types.
"""