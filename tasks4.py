from utilities import *

data_class = Data()

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