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
    fill_speed(get_connection())

if __name__ == "__main__":
    main()