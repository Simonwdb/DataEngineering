import sqlite3
import pandas as pd
from datetime import datetime


class Data:
    def __init__(self):
        self.db_path = "./Data/flights_database.db"
        self.make_connection()

    def make_connection(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

    def get_dataframe(self, query, ):
        self.cursor.execute(query)
        data_rows = self.cursor.fetchall()
        df = pd.DataFrame(data_rows, columns=[col[0] for col in self.cursor.description])
        return df
    
data_class = Data()