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
airports_df = data['airports']

# Dummy data
def generate_dummy_data():
    np.random.seed(42)
    airports = ['JFK', 'LGA', 'EWR', 'SFO', 'LAX', 'ORD', 'ATL', 'MIA']
    airlines = ['Delta', 'American', 'United', 'JetBlue', 'Southwest']
    
    flights = pd.DataFrame({
        'Flight ID': range(1, 201),
        'Departure Airport': np.random.choice(airports[:3], 200),
        'Arrival Airport': np.random.choice(airports[3:], 200),
        'Airline': np.random.choice(airlines, 200),
        'Distance (km)': np.random.randint(300, 5000, 200),
        'Departure Delay (min)': np.random.randint(-10, 120, 200),
        'Arrival Delay (min)': np.random.randint(-10, 150, 200),
        'Taxi Time (min)': np.random.randint(5, 40, 200),
        'Passengers': np.random.randint(50, 300, 200),
        'Departure Time': pd.date_range('2023-01-01', periods=200, freq='h').time,
        'Weather Condition': np.random.choice(['Clear', 'Rain', 'Storm', 'Foggy'], 200)
    })
    return flights

flights_data = generate_dummy_data()


# Streamlit UI
st.set_page_config(page_title='NYC Flight Dashboard', layout='wide')
st.title('✈ NYC Flight Dashboard')

# Sidebar
st.sidebar.header('Navigation')
page = st.sidebar.radio("Go to section:", ['Overview', 'Airport Comparison', 'Delays & Causes', 'Daily Flights', 'Aircraft Types & Speed', 'Weather Impact'])

if page == 'Overview':
    st.header('📊 General Flight Statistics')
    average_total_delay = (flights_df['dep_date_delay'] + flights_df['arr_date_delay']).mean()
    percentage_without_delay = ((flights_df['dep_date_delay'] <= 0) & (flights_df['arr_date_delay'] <= 0)).mean() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Flights", len(flights_df))
    col2.metric("Average Delay", f"{average_total_delay:.1f} min")
    col3.metric("Unique Destinations", flights_df['dest'].nunique())
    col4.metric("Percentage of Flights Without Delay", f"{percentage_without_delay:.1f}%")

    # Add the map with airports
    st.subheader('Airports in the US')
    airports_fig = plot_airports_by_region(data['airports'])
    st.plotly_chart(airports_fig)
    
    fig = px.histogram(flights_data, x='Arrival Airport', title='Top Destinations')
    st.plotly_chart(fig)

elif page == 'Airport Comparison':
    st.header('🏢 Airport Comparison')
    departure = st.selectbox('Select departure airport:', flights_df['dest'].unique())

    # Check if airports_df is empty
    if airports_df[airports_df['faa'] == departure].empty:
        st.warning("No matching FAA codes found in Airports dataset.")
    else:
        airport_data = flights_df[flights_df['dest'] == departure]
        average_airport_delay = (airport_data['dep_date_delay'] + airport_data['arr_date_delay']).mean()
        percentage_airport_delay = ((airport_data['dep_date_delay'] <= 0) & (airport_data['arr_date_delay'] <= 0)).mean() * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"Total Flights to: {airports_df[airports_df['faa'] == departure]['name'].values[0]}", len(airport_data))
        col2.metric("Average Delay", f"{average_airport_delay:.1f} min")
        col3.metric("Average Taxi Time", f"{airport_data['taxi_time'].mean():.1f} min")
        col4.metric("Percentage of Flights Without Delay", f"{percentage_airport_delay:.1f}%")
        
        fig = plot_flight_from_nyc(departure, data['airports'])
        st.plotly_chart(fig)

    # fig = px.box(airport_data, y='Arrival Delay (min)', title='Delays per Airport')
    # st.plotly_chart(fig)

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
