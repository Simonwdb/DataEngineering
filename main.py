from tasks1 import *
from tasks3 import *
import sqlite3

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

def main():
    print("Here we call the functions from the points described in the functions folder/file")
    fill_speed(get_connection())

if __name__ == "__main__":
    main()