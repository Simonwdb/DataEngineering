import streamlit as st
import pandas as pd
import plotly.express as px

# importing the tasks file
from tasks4 import (
    flights_df,
    planes_df,
    airports_df,
    airlines_df,
    avg_speed_df,
    min_date,
    max_date
)
from tasks1 import (
    plot_airports_by_region,
    plot_timezones,
    plot_flight_from_nyc
)
from tasks3 import (
    get_statistics,
    plot_destinations,
    get_plane_model_counts,
    distance_vs_delay,
    plot_top_10_delayed_airlines,
    plot_average_speed_per_manufacturer,
    analyze_inner_product_vs_air_time
)

# Streamlit UI
st.set_page_config(page_title='NYC 2023 Flight Dashboard', layout='wide')
st.title('✈ NYC 2023 Flight Dashboard')

# Sidebar
st.sidebar.header('Navigation')
page = st.sidebar.radio("Go to section:", ['Overview', 'Departure Airport Comparison', 'Arrival Airport Comparison', 'Departure-Arrival Analysis', 'Delays & Causes', 'Daily Flights', 'Aircraft Types & Speed', 'Weather Impact'])

if page == 'Overview':
    st.header('📊 General Flight Statistics')
    average_total_delay = (flights_df['dep_date_delay'] + flights_df['arr_date_delay']).mean()
    percentage_without_delay = ((flights_df['dep_date_delay'] <= 0) & (flights_df['arr_date_delay'] <= 0)).mean() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Flights", len(flights_df))
    col2.metric("Average Delay", f"{average_total_delay:.1f} min")
    col3.metric("Unique Destinations", flights_df['dest'].nunique())
    col4.metric("Flights Without Delay", f"{percentage_without_delay:.1f}%")

    # Add the map with airports
    st.subheader('Airports in the US')
    airports_fig = plot_airports_by_region(airports_df)
    st.plotly_chart(airports_fig)

    # Add the timezone distribution plot
    st.subheader('Distribution of Arrival Airports by Time Zone')
    timezone_fig = plot_timezones(airports_df)
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
        
        fig = plot_flight_from_nyc(departure, airports_df)
        st.plotly_chart(fig)


elif page == 'Departure Airport Comparison':
    st.header('🏢 Departure Airport Comparison')

    # Select departure airport
    departure = st.selectbox('Select departure airport:', flights_df['origin'].unique())
    departure_name = airports_df[airports_df['faa'] == departure]['name'].values[0]



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

elif page == 'Departure-Arrival Analysis':
    st.header('🌍 Departure-Arrival Analysis')

    # Select departure and arrival airport with default values
    origin_options = flights_df['origin'].unique()
    dest_options = flights_df['dest'].unique()

    departure_airport = st.selectbox(
        'Select departure airport:',
        origin_options,
        index=origin_options.tolist().index('JFK') if 'JFK' in origin_options else 0
    )

    arrival_airport = st.selectbox(
        'Select arrival airport:',
        dest_options,
        index=dest_options.tolist().index('ATL') if 'ATL' in dest_options else 0
    )

    # Filter dataset based on selected airports
    route_data = flights_df[(flights_df['origin'] == departure_airport) & (flights_df['dest'] == arrival_airport)].copy()

    if route_data.empty:
        st.warning("No flights found between the selected airports.")
    else:
        # Route statistics
        total_flights = len(route_data)
        avg_delay = route_data['total_delay'].mean()
        top_carrier = route_data['carrier'].mode()[0]
        carrier_lookup = airlines_df.set_index('carrier')['name'].to_dict()
        most_frequent_airline = carrier_lookup.get(top_carrier, top_carrier)


        # Display metrics
        col1, col2, col3 = st.columns([2, 2, 2])
        col1.metric("Total Flights", total_flights)
        col2.metric("Average Delay", f"{avg_delay:.1f} min")
        col3.metric("Most Frequent Airline", most_frequent_airline)

        # Visualization: Flight Volume Over Time
        st.subheader('Flight Volume Over Time')
        route_data.loc[:, 'date'] = pd.to_datetime(route_data['sched_dep_date'])
        daily_flights = route_data.groupby(route_data['date'].dt.date).size().reset_index(name='count')
        time_fig = px.line(daily_flights, x='date', y='count', title="Daily Flight Volume")
        time_fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Number of Flights',
                showlegend=False
            )
        st.plotly_chart(time_fig)

        # Visualization: Average Total Delay Over Time
        st.subheader('Average Total Delay Over Time')
        daily_delay = (
            route_data
            .groupby(route_data['date'].dt.date)['total_delay']
            .mean()
            .reset_index(name='avg_total_delay')
        )
        fig = px.line(
            daily_delay,
            x='date',
            y='avg_total_delay',
            title="Average Total Delay per Day",
            labels={'date': 'Date', 'avg_total_delay': 'Avg Total Delay (min)'}
        )
        st.plotly_chart(fig)

        # Visualization: Top Aircraft Types
        st.subheader("Top Aircraft Types on This Route")
        if 'tailnum' in route_data.columns:
            plane_models = get_plane_model_counts(departure_airport, arrival_airport)
            plane_model_df = pd.DataFrame(list(plane_models.items()), columns=['Plane Model', 'Count'])
            plane_model_df = plane_model_df.nlargest(10, 'Count')
            fig = px.bar(plane_model_df, x='Plane Model', y='Count', text='Count', title='Top Aircraft by Number of Flights')
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig)
        else:
            st.info("Aircraft type data is not available for this route.")

   # Visualization: Top Airlines
    st.subheader("Top Airlines on This Route")
    if 'carrier' in route_data.columns:
        airline_counts = route_data['carrier'].value_counts().reset_index()
        airline_counts.columns = ['carrier', 'Count']

        airline_counts = pd.merge(
            airline_counts,
            airlines_df,  
            on='carrier',
            how='left'
        )

        airline_counts = airline_counts.nlargest(10, 'Count')

        fig = px.bar(
            airline_counts,
            x='name',    
            y='Count',
            text='Count',
            title='Top Airlines by Number of Flights'
        )

        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis_title='Airline',
            yaxis_title='Number of Flights',
            showlegend=False
        )
        st.plotly_chart(fig)
    else:
        st.info("Airline data is not available for this route.")

elif page == 'Delays & Causes':
    st.header('⏳ Delays & Causes')
    
    st.subheader('Correlation between Distance vs Arrival Delay')
    corr_fig = distance_vs_delay()
    st.plotly_chart(corr_fig)
       
    # Add the top 10 delayed airlines chart
    st.subheader('Top 10 Airlines with the Most Delays')
    fig = plot_top_10_delayed_airlines(flights_df, airlines_df)
    st.plotly_chart(fig)


elif page == 'Daily Flights':
    st.header('📅 Flights on a Specific Day')
    # Date input with a calendar widget
    selected_date = st.date_input(
        'Select a date:',
        min_value=min_date,
        max_value=max_date,
        value=min_date  # Default to the earliest date
    )
    
    day_df = flights_df[flights_df['sched_dep_date'].dt.date== selected_date]
    day_destinations = day_df['dest'].unique()
    day_num_destinations = day_df['dest'].nunique()

    # Calculate the amount of passengers that can travel on selected_date
    day_df = day_df.merge(planes_df[['tailnum', 'seats']], on='tailnum', how='left')
    passenger_per_dest_df = day_df.groupby('dest')['seats'].sum().reset_index()
    passenger_per_dest_df = passenger_per_dest_df.sort_values(by='seats', ascending=False)
    passenger_amount = passenger_per_dest_df['seats'].sum()
    flights_amount = day_df['flight'].nunique()

    # Find the most popular destination (the one with the most passengers)
    favorite_destination = passenger_per_dest_df.iloc[0]['dest'] if not passenger_per_dest_df.empty else "N/A"
    favorite_destination_passengers = passenger_per_dest_df.iloc[0]['seats'] if not passenger_per_dest_df.empty else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Flights", flights_amount)
    col2.metric("Total Passengers", int(passenger_amount))
    col3.metric("Unique Destinations", day_num_destinations)
    col4.metric("Top Destination", favorite_destination)
    col5.metric("Passengers to Top Destination", int(favorite_destination_passengers))

    # Plot the map with the multiple destinations
    st.subheader(f"Destinations from NYC on {selected_date}")
    fig = plot_flight_from_nyc(day_destinations, airports_df)
    st.plotly_chart(fig)

    # Add summary stats below
    st.subheader("📊 Summary Statistics")
    avg_dep_delay = day_df['dep_date_delay'].mean()
    avg_arr_delay = day_df['arr_date_delay'].mean()
    avg_taxi = day_df['taxi_time'].mean()
    total_passengers = day_df['seats'].sum()
 
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Departure Delay", f"{avg_dep_delay:.1f} min")
    col2.metric("Avg Arrival Delay", f"{avg_arr_delay:.1f} min")
    col3.metric("Avg Taxi Time", f"{avg_taxi:.1f} min")
    col4.metric("Total Passengers", total_passengers)

elif page == 'Aircraft Types & Speed':
    st.header('🚀 Aircraft Types & Speed')
    
    fig = plot_average_speed_per_manufacturer(avg_speed_df)
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
    

st.sidebar.write('Created for the analysis of flights from NYC')
