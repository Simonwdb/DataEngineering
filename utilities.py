import math
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from datetime import datetime


class Data:
    def __init__(self):
        self.db_path = "./Data/flights_database.db"
        self.make_connection()

    def make_connection(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def get_dataframe(self, query, ):
        self.cursor.execute(query)
        data_rows = self.cursor.fetchall()
        df = pd.DataFrame(data_rows, columns=[col[0] for col in self.cursor.description])
        return df