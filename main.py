from tasks3 import *
import sqlite3
import pandas as pd

def get_columns_from_table(conn, table_name):
    """
    Returns a list of column names from a given table.
    """
    query = f"SELECT * FROM {table_name} LIMIT 1"
    df = pd.read_sql_query(query, conn)
    return df.columns.tolist()

#loading the data
def get_connection(db_path="./Data/flights_database.db"):
    conn = sqlite3.connect(db_path)
    print("Connection to SQLite DB successful")
    return conn

def main():
    print("Here we call the functions from the points described in the functions folder/file")
    ATL = distance_vs_delay(get_connection())

    print(ATL)

if __name__ == "__main__":
    main()