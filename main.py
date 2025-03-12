from utilities import *
from tasks4 import *

data_class = Data()

# Get the general dataframes from the database
query_flights = '''SELECT * from flights'''
flights_df = data_class.get_dataframe(query_flights)

query_airports = '''SELECT * from airports'''
airports_df = data_class.get_dataframe(query_airports)

query_planes = '''SELECT * from planes'''
planes_df = data_class.get_dataframe(query_planes)

query_airlines = '''SELECT * from airlines'''
airlines_df = data_class.get_dataframe(query_airlines)

query_weather = '''SELECT * from weather'''
weather_df = data_class.get_dataframe(query_weather)

def get_dataframe_safe(data_class, query):
    try:
        return data_class.get_dataframe(query)
    except Exception as e:
        return pd.DataFrame()  # Return an empty DataFrame to prevent crashes

def load_data():
    """Load all necessary data from the database."""
    data_class = Data()
    
    queries = {
        "flights": "SELECT * FROM flights",
        "airports": "SELECT * FROM airports",
        "planes": "SELECT * FROM planes",
        "airlines": "SELECT * FROM airlines",
        "weather": "SELECT * FROM weather"
    }
    
    dataframes = {name: get_dataframe_safe(data_class, query) for name, query in queries.items()}
    
    return dataframes

def process_flights_data(flights_df, airports_df):
    if flights_df.empty:
        return flights_df
    
    flights_df = remove_nan_values(flights_df)
    flights_df = remove_duplicates(flights_df)
    flights_df = convert_time_columns(flights_df)
    flights_df = adjust_flight_dates(flights_df)
    flights_df = calculate_delays(flights_df)
    flights_df = adjust_negative_delays(flights_df)
    flights_df = check_delay_equality(flights_df)
    flights_df = merge_timezone_info(flights_df, airports_df)
    flights_df = convert_arr_date_to_gmt5(flights_df)
    flights_df = calculate_block_and_taxi_time(flights_df)
    
    return flights_df

# Perform some data wrangling on the flights_df
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
# After this block the flights_df is ready to use as visualization or stastical dataframe 


def main():
    print("Here we call the functions from the points described in the functions folder/file")
    # fill_speed(get_connection())

if __name__ == "__main__":
    main()