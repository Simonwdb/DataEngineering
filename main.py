from utilities import *
from tasks4 import *

data_class = Data()

'''

POSSIBLE REDUNDANT CODE:

#loading the data
def get_connection(db_path="./Data/flights_database.db"):
    conn = sqlite3.connect(db_path)
    print("Connection to SQLite DB successful")
    return conn

# making the conn as a global variable
conn = get_connection()

# Returning a dataframe from a given query
def get_dataframe(query):
    cursor = conn.cursor()
    cursor.execute(query)
    data_rows = cursor.fetchall()
    df = pd.DataFrame(data_rows, columns=[col[0] for col in cursor.description])
    return df

'''

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



def main():
    print("Here we call the functions from the points described in the functions folder/file")
    # fill_speed(get_connection())

if __name__ == "__main__":
    main()