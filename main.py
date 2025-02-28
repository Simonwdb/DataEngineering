from tasks1 import *
from tasks3 import *
import sqlite3

#loading the data
def get_connection(db_path="./Data/flights_database.db"):
    conn = sqlite3.connect(db_path)
    print("Connection to SQLite DB successful")
    return conn

def main():
    print("Here we call the functions from the points described in the functions folder/file")
    ATL = distance_vs_delay(get_connection())
    print(ATL)

    plot_airport_routes_plotly(["AAF", "AAP"])
    plot_airport_routes_plotly(["TZR", "AAP", "AAF"])

if __name__ == "__main__":
    main()