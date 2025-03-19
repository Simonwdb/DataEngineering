"""

For now I created a example dashboard with dummy data. To see how it looks, and works. When this works, we can use our data in main.py.
Best to make main.py into sort like file as this example.

Notes:
- Add a plot that shows the graph between the distance from the origin to the destination.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# importing the tasks file
from utilities import *
from tasks4 import *
from tasks1 import *
from tasks3 import *

# creating the data class
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

# prepare the datasets
data = load_data()

flights_df = process_flights_data(data['flights'], data['airports'])



# Streamlit UI
st.set_page_config(page_title='NYC Flight Dashboard', layout='wide')
st.title('✈ NYC Flight Dashboard')

# Sidebar
st.sidebar.header('Navigation')
page = st.sidebar.radio("Go to section:", ['Overview', 'Airport Comparison', 'Delays & Causes', 'Daily Flights', 'Aircraft Types & Speed', 'Weather Impact'])

if page == 'Overview':
    st.header('📊 General Flight Statistics')
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Flights", len(flights_data))
    col2.metric("Average Delay", f"{flights_data['Arrival Delay (min)'].mean():.1f} min")
    col3.metric("Unique Destinations", flights_data['Arrival Airport'].nunique())
    col4.metric("Percentage of Flights Without Delay", f"{(flights_data['Arrival Delay (min)'] <= 0).mean() * 100:.1f}%")
    
    fig = px.histogram(flights_data, x='Arrival Airport', title='Top Destinations')
    st.plotly_chart(fig)

elif page == 'Airport Comparison':
    st.header('🏢 Airport Comparison')
    departure = st.selectbox('Select departure airport:', flights_data['Departure Airport'].unique())
    
    airport_data = flights_data[flights_data['Departure Airport'] == departure]
    st.metric("Average Delay", f"{airport_data['Arrival Delay (min)'].mean():.1f} min")
    st.metric("Average Taxi Time", f"{airport_data['Taxi Time (min)'].mean():.1f} min")
    
    fig = px.box(airport_data, y='Arrival Delay (min)', title='Delays per Airport')
    st.plotly_chart(fig)

elif page == 'Delays & Causes':
    st.header('⏳ Delays & Causes')
    
    fig = px.scatter(flights_data, x='Distance (km)', y='Arrival Delay (min)', title='Delay vs Distance')
    st.plotly_chart(fig)
    
    fig = px.box(flights_data, x='Weather Condition', y='Arrival Delay (min)', title='Delay vs Weather Conditions')
    st.plotly_chart(fig)
    
    airline_delay = flights_data.groupby('Airline')['Arrival Delay (min)'].mean().reset_index()
    fig = px.bar(airline_delay, x='Airline', y='Arrival Delay (min)', title='Average Delay per Airline')
    st.plotly_chart(fig)

elif page == 'Daily Flights':
    st.header('📅 Flights on a Specific Day')
    date = st.date_input('Select a date', pd.to_datetime('2023-01-01'))
    
    day_flights = flights_data.sample(10) 
    st.write(day_flights)
    
    fig = px.scatter_geo(day_flights, locations='Arrival Airport', title='Destinations of the Day')
    st.plotly_chart(fig)

elif page == 'Aircraft Types & Speed':
    st.header('🚀 Aircraft Types & Speed')
    
    flights_data['Speed (km/h)'] = flights_data['Distance (km)'] / (np.random.randint(1, 6, len(flights_data)))
    fig = px.histogram(flights_data, x='Speed (km/h)', title='Average Speed per Aircraft')
    st.plotly_chart(fig)

elif page == 'Weather Impact':
    st.header('🌦️ Weather Impact on Flights')
    
    weather_impact = flights_data.groupby('Weather Condition')['Arrival Delay (min)'].mean().reset_index()
    fig = px.bar(weather_impact, x='Weather Condition', y='Arrival Delay (min)', title='Average Delay per Weather Type')
    st.plotly_chart(fig)

st.sidebar.write('Created for the analysis of NYC flights')
