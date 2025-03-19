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
page = st.sidebar.radio("Go to section:", ['Overview', 'Departure Airport Comparison', 'Arrival Airport Comparison', 'Delays & Causes', 'Daily Flights', 'Aircraft Types & Speed', 'Weather Impact'])

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

    # Add the timezone distribution plot
    st.subheader('Distribution of Airports by Time Zone')
    timezone_fig = plot_timezones(data['airports'])
    st.plotly_chart(timezone_fig)

    # Get the top 10 destinations based on frequency and sort them
    top_10_destinations = flights_df['dest'].value_counts().nlargest(10).sort_values(ascending=False)

    # Create a DataFrame for the top 10 destinations
    top_10_df = pd.DataFrame({'dest': top_10_destinations.index, 'count': top_10_destinations.values})

    # Create the histogram for the top 10 destinations
    fig = px.bar(top_10_df, x='dest', y='count', title='Top 10 Destinations', text='count')

    # Update the layout to show the count values above the bars
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title='Destination',
        yaxis_title='Number of Flights',
        showlegend=False,
        height=600,
        width=1000
    )

    # Display the chart in the dashboard
    st.plotly_chart(fig)

elif page == 'Arrival Airport Comparison':
    st.header('🏢 Arrival Airport Comparison')
    departure = st.selectbox('Select arrival airport:', flights_df['dest'].unique())

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

elif page == 'Departure Airport Comparison':
    st.header('🏢 Departure Airport Comparison')

    # Select departure airport
    departure = st.selectbox('Select departure airport:', flights_df['origin'].unique())
    departure_name = airports_df[airports_df['faa'] == departure]['name'].values[0]

    # Determine the minimum and maximum dates from dep_date and arr_date columns
    min_date = pd.to_datetime(flights_df['sched_dep_date'].min()).date()
    max_date = pd.to_datetime(flights_df['arr_date'].max()).date()

    # Date input with a calendar widget
    selected_date = st.date_input(
        'Select a date:',
        min_value=min_date,
        max_value=max_date,
        value=min_date  # Default to the earliest date
    )

    # Extract month and day from the selected date
    month = selected_date.month
    day = selected_date.day

    # Get statistics for the selected day and airport
    stats = get_statistics(month, day, departure, flights_df)

    if not stats:  # Check if stats is empty (no flights found)
        st.warning(f"No flights found on {month}/{day} from {departure_name}.")
    else:
        # Display statistics
        st.subheader(f"Statistics for flights on {selected_date} from {departure_name}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Flights", stats['num_flights'])
        col2.metric("Unique Destinations", stats['num_unique_destinations'])
        col3.metric("Most Frequent Destination", stats['most_frequent_destination'])
        col4.metric("Flights to Most Frequent Destination", stats['most_frequent_destination_count'])

        # Optionally, add a map or other visualizations here
        st.subheader(f"Map of flights from {departure_name} on {selected_date}")
        fig = plot_destinations(month=month, day=day, origin_airport=departure, flights_df=flights_df, airports_df=airports_df)
        fig.update_layout(
            height=600,
            width=1000
        )
        st.plotly_chart(fig)

        # Add the bar chart for top 10 plane models
        frequent_airport_name = airports_df[airports_df['faa'] == stats['most_frequent_destination']]['name'].values[0]
        st.subheader(f"Top 10 Plane Models from {departure_name} to {frequent_airport_name} on {selected_date}")
        
        # Get the top 10 plane models for the selected route
        plane_model_counts = get_plane_model_counts(departure, stats['most_frequent_destination'])
        plane_model_df = pd.DataFrame(list(plane_model_counts.items()), columns=['plane_model', 'count'])
        plane_model_df = plane_model_df.nlargest(10, 'count')  # Get the top 10

        # Create the bar chart
        fig = px.bar(
            plane_model_df,
            x='plane_model',
            y='count',
            text='count'
        )

        # Update the layout to match the style of the Top 10 Destinations chart
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis_title='Plane Model',
            yaxis_title='Number of Flights',
            showlegend=False,
            height=600,
            width=1000
        )

        # Display the chart
        st.plotly_chart(fig)

elif page == 'Delays & Causes':
    st.header('⏳ Delays & Causes')
    
    st.subheader('Correlation between Distance vs Arrival Delay')
    corr_fig = distance_vs_delay()
    st.plotly_chart(corr_fig)
    
    # Add the enhanced boxplot for delays vs visibility by origin
    st.subheader('Flight Delays vs Visibility Conditions by Origin Airport')
    fig = plot_delay_vs_visibility(flights_df, data['weather'])
    st.plotly_chart(fig)
    
    # Add the top 10 delayed airlines chart
    st.subheader('Top 10 Airlines with the Most Delays')
    fig = plot_top_10_delayed_airlines(flights_df, data['airlines'])
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
    st.header('🌬️ Weather Impact on Air Time')
    
    df, fig = analyze_inner_product_vs_air_time()
    
    if fig is not None:
        st.plotly_chart(fig)
        
        st.subheader("Average Air Time by Inner Product Sign")
        
        col1, col2, col3 = st.columns(3)
        summary = df.groupby('ip_sign')['air_time'].mean()
        col1.metric("Negative Inner Product", f"{summary['negative']:.1f}")
        col2.metric("Positive Inner Product", f"{summary['positive']:.1f}")
        col3.metric("Zero Inner Product", f"{summary['zero']:.1f}")
    
    else:
        st.warning("No valid flight records found for analysis.")
    

st.sidebar.write('Created for the analysis of NYC flights')
