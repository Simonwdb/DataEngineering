from utilities import *
from tasks4 import *
from tasks1 import *
from tasks3 import *

data_class = Data()

def get_dataframe_safe(data_class, query):
    try:
        return data_class.get_dataframe(query)
    except Exception as e:
        return pd.DataFrame()

def load_data():    
    queries = {
        "flights": "SELECT * FROM flights",
        "airports": "SELECT * FROM airports",
        "planes": "SELECT * FROM planes",
        "airlines": "SELECT * FROM airlines",
        "weather": "SELECT * FROM weather"
    }
    
    fill_speed() # update the planes database

    dataframes = {name: get_dataframe_safe(data_class, query) for name, query in queries.items()}
    
    return dataframes

def process_flights_data(flights_df, airports_df):
    if flights_df.empty:
        return flights_df
    
    flights_df = remove_nan_values(flights_df)
    remove_duplicates(flights_df)
    convert_time_columns(flights_df)
    adjust_flight_dates(flights_df)
    calculate_delays(flights_df)
    adjust_negative_delays(flights_df)
    check_delay_equality(flights_df)
    flights_df = merge_timezone_info(flights_df, airports_df)
    convert_arr_date_to_gmt5(flights_df)
    calculate_block_and_taxi_time(flights_df)
    
    return flights_df

data = load_data()

flights_df = process_flights_data(data['flights'], data['airports'])
airpors_df = data['airports']