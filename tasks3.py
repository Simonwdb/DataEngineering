import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import math

from utilities import data_class

"""Verify that the distances you computed in part 1 are roughly equal to the dis-
tances in the variable distance in the table flights. If they are much oﬀ,
recall that latitude and longitude represent angles expressed in degrees, while the
functions sin and cos expects entries in radial angles."""

def verify_distances(flights_df, airports_df):
    """
    Verifies that the distances computed in part 1 (geodetisch vanaf JFK) are roughly equal
    to the recorded flight distances in the flights table.
    
    Assumes that:
      - airports_df bevat ten minste de kolommen: 'faa', 'lat', 'lon'
      - flights_df bevat ten minste de kolommen: 'dest' (destination airport code) en 'distance' (recorded flight distance in km)
      
    Returns:
        dict: Een dictionary met:
              - 'mean_difference': het gemiddelde verschil (recorded - computed) in kilometers,
              - 'correlation': de correlatie tussen de recorded en computed distances.
    """
    # Zoek naar JFK in de airports DataFrame (gebruik kolom 'faa')
    jfk_airport = airports_df[airports_df['faa'] == 'JFK']
    if jfk_airport.empty:
        raise ValueError("JFK airport not found in the airports DataFrame!")
    jfk_airport = jfk_airport.iloc[0]
    jfk_lat_deg = jfk_airport['lat']
    jfk_lon_deg = jfk_airport['lon']
    
    # Aardstraal in kilometers
    R = 6371
    
    # Werk op een kopie zodat de originele DataFrame niet wordt aangepast
    airports_copy = airports_df.copy()
    
    # Converteer breedte- en lengtegraad van alle luchthavens naar radialen
    airports_copy['lat_rad'] = np.radians(airports_copy['lat'])
    airports_copy['lon_rad'] = np.radians(airports_copy['lon'])
    jfk_lat_rad = np.radians(jfk_lat_deg)
    jfk_lon_rad = np.radians(jfk_lon_deg)
    
    # Bereken de verschillen in radialen
    dphi = airports_copy['lat_rad'] - jfk_lat_rad      # Δφ
    dlambda = airports_copy['lon_rad'] - jfk_lon_rad    # Δλ
    phi_m = (airports_copy['lat_rad'] + jfk_lat_rad) / 2  # Middenwaarde van de breedtegraad
    
    # Bereken de geodetische afstand volgens de formule:
    # distance = R * sqrt((2*sin(Δφ/2)*cos(Δλ/2))^2 + (2*cos(φ_m)*sin(Δλ/2))^2)
    airports_copy['computed_distance_km'] = R * np.sqrt(
        (2 * np.sin(dphi / 2) * np.cos(dlambda / 2))**2 +
        (2 * np.cos(phi_m) * np.sin(dlambda / 2))**2
    )
    
    # Merge de computed distance met de flights DataFrame
    # We gaan ervan uit dat in flights_df de kolom 'dest' staat met de bestemming (FAA-code)
    merged_df = flights_df.merge(airports_copy[['faa', 'computed_distance_km']],
                                 left_on='dest', right_on='faa', how='left')
    
    # Bereken het verschil tussen de recorded afstand (flights_df['distance']) en de computed afstand
    diff = merged_df['distance'] - merged_df['computed_distance_km']
    
    mean_diff = diff.mean()
    correlation = merged_df['distance'].corr(merged_df['computed_distance_km'])
    
    return {
        "mean_difference": mean_diff,
        "correlation": correlation
    }

"""For each flight, the origin from which it leaves can be found in the variable
origin in the table flights. Identify all diﬀerent airports in NYC from
which flights depart and save a DataFrame contain the information about those
airports from airports"""
def airports_in_nyc(flights_df, airports_df):
    # Get the airports in NYC from the flights table
    # Get the information about those airports from the airports table
    # Save the information in a DataFrame
    return airports_df[airports_df['faa'].isin(flights_df['origin'].unique())]

"""Write a function that takes a month and day and an airport in NYC as input,
and produces a figure similar to the one from part 1 containing all destinations
of flights on that day."""
def plot_destinations(month, day, origin_airport, flights_df, airports_df):
    """
    Produces a figure containing all destination airports for flights on a given month and day
    from the specified NYC airport.
    
    Parameters:
        month (int): Month number (e.g., 1 for January).
        day (int): Day of the month.
        origin_airport (str): FAA code of the NYC origin airport (e.g., 'JFK', 'LGA', 'EWR').
    
    Returns:
        fig (plotly.graph_objs._figure.Figure): A Plotly figure containing the destination airports.
    """
    # Filter flights for the specified day and origin
    flights_on_day = flights_df[
        (flights_df['month'] == month) &
        (flights_df['day'] == day) &
        (flights_df['origin'] == origin_airport)
    ]
    
    if flights_on_day.empty:
        print(f"No flights found on {month}/{day} from {origin_airport}.")
        return None
    
    # Get unique destination codes from these flights
    dest_codes = flights_on_day['dest'].unique()
    
    # Filter the airports DataFrame to get information on these destination airports
    dest_airports = airports_df[airports_df['faa'].isin(dest_codes)].copy()
    
    # (Optional) Count the number of flights per destination for marker sizing
    flight_counts = flights_on_day['dest'].value_counts().rename_axis('faa').reset_index(name='count')
    dest_airports = dest_airports.merge(flight_counts, on='faa', how='left')
    
    # Create a scatter_geo figure (returning the figure object without calling fig.show())
    fig = px.scatter_geo(dest_airports,
                         lat="lat", lon="lon",
                         hover_name="name",
                         size="count",
                         title=f"Destinations from {airports_df[airports_df['faa'] == origin_airport]['name'].values[0]} on {month}/{day}",
                         projection="albers usa")
    
    return fig
    
"""Also write a function that returns statistics for that day, i.e. how many flights,
how many unique destinations, which destination is visited most often, etc."""
def get_statistics(month, day, airport, flights_df):
    """
    Returns statistics for flights departing from a given NYC airport on a specified day.
    
    Parameters:
        month (int): The month (e.g., 6 for June).
        day (int): The day of the month.
        airport (str): The FAA code for the NYC origin airport (e.g., 'JFK', 'LGA', 'EWR').
    
    Returns:
        dict: A dictionary containing:
              - 'num_flights': Total number of flights on that day.
              - 'num_unique_destinations': Number of unique destination airports.
              - 'most_frequent_destination': The destination visited most often.
              - 'most_frequent_destination_count': Number of flights to the most frequent destination.
    """
    # Filter the flights for the specified day and airport
    flights_on_day = flights_df[
        (flights_df['month'] == month) &
        (flights_df['day'] == day) &
        (flights_df['origin'] == airport)
    ]
    
    if flights_on_day.empty:
        print(f"No flights found on {month}/{day} from {airport}.")
        return {}
    
    # Calculate statistics
    num_flights = len(flights_on_day)
    num_unique_destinations = flights_on_day['dest'].nunique()
    
    # Identify the most frequent destination
    dest_counts = flights_on_day['dest'].value_counts()
    most_frequent_destination = dest_counts.idxmax()
    most_frequent_destination_count = dest_counts.max()
    
    stats = {
        "num_flights": num_flights,
        "num_unique_destinations": num_unique_destinations,
        "most_frequent_destination": most_frequent_destination,
        "most_frequent_destination_count": most_frequent_destination_count
    }
    
    return stats

"""Write a function that, given a departing airport and an arriving airport, returns
a dict describing how many times each plane type was used for that flight
trajectory. For this task you will need to match the columns tailnum to type
in the table planes and match this to the tailnum s in the table flights."""
def get_plane_types(origin, destination):
    # Get the plane types for the flight trajectory
    # Return the dict
    query_type_counts = '''
    SELECT p.type, COUNT(*) as count 
    FROM flights as f
    JOIN planes as p ON f.tailnum = p.tailnum
    WHERE f.origin = ? AND f.dest = ?
    GROUP BY p.type
    ORDER BY count DESC
    '''

    cursor = data_class.cursor
    cursor.execute(query_type_counts, (origin, destination))
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=[x[0] for x in cursor.description])
    result_dict = dict(zip(df['type'], df['count']))

    return result_dict

def get_plane_model_counts(origin, destination):
    query_model_counts = '''
    SELECT p.manufacturer || ' ' || p.model AS plane_model, COUNT(*) as count
    FROM flights AS f
    JOIN planes AS p ON f.tailnum = p.tailnum
    WHERE f.origin = ? AND f.dest = ?
    GROUP BY plane_model
    ORDER BY count DESC
    '''
    cursor = data_class.cursor
    cursor.execute(query_model_counts, (origin, destination))
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=[x[0] for x in cursor.description])
    result_dict = dict(zip(df['plane_model'], df['count']))
    
    return result_dict

"""Compute the average departure delay per flight for each of the airlines. Visualize
the results in a barplot with the full (rotated) names of the airlines on the x-axis."""
def average_delay_per_airline(flights_df):
    # Compute the average departure delay per flight for each airline
    # Visualize the results in a barplot
    avg_delay_df = flights_df.groupby('carrier', as_index=False)['dep_delay'].mean()
    avg_delay_df.rename(columns={'dep_delay': 'avg_dep_delay'}, inplace=True)
    avg_delay_df['avg_dep_delay'] = avg_delay_df['avg_dep_delay'].round(2)
    
    fig = px.bar(
        avg_delay_df,
        x='carrier',
        y='avg_dep_delay',
        title='Average Departure Delay per Airline',
        labels={'avg_dep_delay': 'Average Departure Delay (minutes)', 'carrier': 'Airline'},
    )

    fig.update_layout(
        xaxis={'categoryorder':'total descending'},
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    # ASK: do we need to return the figure or do we need to use fig.show() in this function?
    return fig

"""Write a function that takes as input a range of months and a destination and
returns the amount of delayed flights to that destination."""
def get_delayed_flights(flights_df, month, destination):
    # Get the amount of delayed flights to the destination
    # Return the amount
    delayed_flights = flights_df[
        flights_df['month'] == month & 
        (flights_df['dest'] == destination) & 
        (flights_df['arr_delay'] > 0)
    ]
    return len(delayed_flights)

def plot_top_10_delayed_airlines(flights_df, airlines_df):
    # Calculate total delay for each flight
    flights_df['total_delay'] = flights_df['dep_date_delay'] + flights_df['arr_date_delay']

    # Group by airline and calculate the average total delay
    airline_delays = flights_df.groupby('carrier')['total_delay'].mean().reset_index()

    # Round the average delay to 1 decimal place
    airline_delays['total_delay'] = airline_delays['total_delay'].round(1)

    # Merge with airlines_df to get full airline names
    airline_delays = airline_delays.merge(airlines_df, left_on='carrier', right_on='carrier', how='left')

    # Sort by average delay in descending order and take the top 10
    top_10_airlines = airline_delays.sort_values('total_delay', ascending=False).head(10)

    # Create the bar chart
    fig = px.bar(
        top_10_airlines,
        x='name',  # Use the full airline name
        y='total_delay',
        title='Top 10 Airlines with the Most Delays',
        labels={'name': 'Airline', 'total_delay': 'Average Total Delay (minutes)'},
        text_auto=True
    )

    # Update layout for better readability
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title='Airline',
        yaxis_title='Average Total Delay (minutes)',
        showlegend=False,
        height=600,
        width=1000
    )

    return fig

"""Write a function that takes a destination airport as input and returns the top 5
airplane manufacturers with planes departing to this destination. For this task,
you have to combine data from flights and planes."""
def top_manufacturers(destination):
    # Get the top 5 airplane manufacturers with planes departing to the destination
    # Return the top 5
    query = """
    SELECT p.manufacturer, COUNT(*) as count
    FROM flights as f
    JOIN planes as p ON f.tailnum = p.tailnum
    WHERE f.dest = ?
    GROUP BY p.manufacturer
    ORDER BY count DESC
    LIMIT 5"""
    
    conn = data_class.conn
    df = pd.read_sql_query(query, conn, params=[destination])
    return df

"""Investigate whether there is a relationship between the distance of a flight and
the arrival delay time."""
def distance_vs_delay():
    conn = data_class.conn
    
    # Query to get distance and arrival delay
    query = """SELECT distance, arr_delay FROM flights WHERE arr_delay IS NOT NULL"""
    df = pd.read_sql_query(query, conn)

    # Calculate the correlation
    correlation = df['distance'].corr(df['arr_delay'])

    # Create a Plotly scatter plot
    fig = px.scatter(
        df,
        x='distance',
        y='arr_delay',
        title=f"Distance vs. Arrival Delay (corr = {correlation:.2f})",
        labels={'distance': 'Distance (km)', 'arr_delay': 'Arrival Delay (minutes)'},
        opacity=0.5
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="Distance (km)",
        yaxis_title="Arrival Delay (minutes)",
        showlegend=False
    )

    return fig

def plot_delay_vs_visibility(merged_df):
    # Define the desired order for the x-axis
    category_order = ['Good (5-10 miles)', 'Moderate (3-5 miles)', 'Poor (1-3 miles)', 'Very Poor (0-1 miles)']

    # Create the boxplot without distinguishing by origin
    fig = px.box(
        merged_df,
        x='visibility_category',
        y='total_delay',
        title='Flight Delays vs Visibility Conditions',
        labels={'visibility_category': 'Visibility', 'total_delay': 'Total Delay (minutes)'},
        category_orders={'visibility_category': category_order},
        color_discrete_sequence=['blue']  # Optioneel: kies een specifieke kleur
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title='Visibility Conditions',
        yaxis_title='Total Delay (minutes)',
        showlegend=False,  # Legenda uitschakelen
        height=600,
        width=1000,
        template='plotly_white'
    )

    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)

    return fig

"""Group the table flights by plane model using the tailnum. For each model,
compute the average speed by taking the average of the distance divided by flight
time over all flights of that model. Use this information to fill the column speed
in the table planes."""


def fill_speed():
    """
    Computes the average speed for each plane model using flights data and
    updates the 'speed' column in the planes table.
    """
    query = """
    SELECT f.tailnum, p.model, AVG(f.distance * 1.0 / (f.air_time / 60)) AS avg_speed
    FROM flights f
    JOIN planes p ON f.tailnum = p.tailnum
    WHERE f.air_time > 0
    GROUP BY f.tailnum, p.model
    """

    conn = data_class.conn
    cursor = data_class.cursor
    cursor.execute(query)
    speeds = cursor.fetchall()  # List of (tailnum, model, avg_speed)

    # Update the planes table with computed speed
    update_query = """
    UPDATE planes
    SET speed = ?
    WHERE tailnum = ?
    """

    for tailnum, model, avg_speed in speeds:
        cursor.execute(update_query, (avg_speed, tailnum))

    conn.commit()  # Save changes to database

def plot_average_speed_per_manufacturer(avg_speed_df):
    fig = px.bar(
        avg_speed_df,
        x='manufacturer',
        y='avg_speed',
        title='Average Speed per Aircraft Manufacturer',
        labels={'avg_speed': 'Average Speed (km/h)', 'manufacturer': 'Manufacturer'},
        text_auto=True
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title='Manufacturer',
        yaxis_title='Average Speed (km/h)',
        showlegend=False,
        height=600,
        width=1000
    )
    return fig

"""The wind direction is given in weather in degrees. Compute for each airport
the direction the plane follows when flying there from New York."""

def calculate_bearing_vectorized(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    delta_lon = lon2 - lon1
    
    x = np.sin(delta_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    
    initial_bearing = np.arctan2(x, y)
    # Convert bearing from radians to degrees and normalize to 0-360
    bearing_degrees = (np.degrees(initial_bearing) + 360) % 360
    
    return bearing_degrees

def average_bearing_vectorized(bearings):
    bearings_rad = np.radians(bearings)
    sin_sum = np.sum(np.sin(bearings_rad))
    cos_sum = np.sum(np.cos(bearings_rad))
    avg_angle_rad = np.arctan2(sin_sum, cos_sum)
    avg_angle_deg = (np.degrees(avg_angle_rad) + 360) % 360
    
    return avg_angle_deg


def calculate_average_bearings_optimized():
    cursor = data_class.cursor

    # Query to get origin, destination, and their coordinates, along with airport names
    query = """
    SELECT f.origin, f.dest, o.lat as origin_lat, o.lon as origin_lon, d.lat as dest_lat, d.lon as dest_lon, d.name as dest_name
    FROM flights f
    JOIN airports o ON f.origin = o.faa
    JOIN airports d ON f.dest = d.faa;
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    # Convert rows to a Pandas DataFrame for efficient processing
    df = pd.DataFrame(rows, columns=['origin', 'dest', 'origin_lat', 'origin_lon', 'dest_lat', 'dest_lon', 'dest_name'])

    # Calculate bearings for all flights at once using vectorized operations
    df['bearing'] = calculate_bearing_vectorized(
        df['origin_lat'], df['origin_lon'], df['dest_lat'], df['dest_lon']
    )

    # Group by destination and calculate average bearings
    average_bearings = df.groupby('dest')['bearing'].apply(average_bearing_vectorized).to_dict()

    # Create a dictionary with destination FAA codes, names, and average bearings
    result = {
        dest: {
            "name": df[df['dest'] == dest]['dest_name'].iloc[0],  # Get the name from the first occurrence
            "average_bearing": avg_bearing
        }
        for dest, avg_bearing in average_bearings.items()
    }

    return result

def plot_bearings_compass(average_bearings):
    # Prepare data for the compass plot
    bearings = [info['average_bearing'] for info in average_bearings.values()]
    destinations = [f"{dest} - {info['name']}" for dest, info in average_bearings.items()]
    
    # Create a polar plot (compass)
    fig = go.Figure()

    for bearing, dest in zip(bearings, destinations):
        fig.add_trace(go.Scatterpolar(
            r=[1],  # Radial distance (constant for all bearings)
            theta=[bearing],  # Angle in degrees
            mode='markers+text',
            marker=dict(size=10, color='blue'),
            text=[dest],
            textposition='top center',
            name=dest
        ))

    # Update layout for the compass plot
    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                rotation=90,  # Rotate the compass so 0° is at the top
                direction='clockwise',  # Bearings are typically measured clockwise
                thetaunit='degrees'
            ),
            radialaxis=dict(visible=False)  # Hide the radial axis
        ),
        title="Average Bearings from NYC Airports to Destinations",
        showlegend=True
    )

    return fig

"""Write a function that computes the inner product between the flight direction
and the wind speed of a given flight."""

def get_flight_record_by_index(index):
    cursor = data_class.cursor
    query = "SELECT * FROM flights LIMIT 1 OFFSET ?"
    cursor.execute(query, (index,))
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"No flight found at position {index}.")
    columns = [desc[0] for desc in cursor.description]
    flight_record = dict(zip(columns, row))
    return flight_record

"""Is there a relation between the sign of this inner product and the air time?"""

def flight_wind_inner_product(flight_record):
    cursor = data_class.cursor
    
    # Get origin coordinates.
    cursor.execute("SELECT lat, lon FROM airports WHERE faa = ?", (flight_record['origin'],))
    origin_coords = cursor.fetchone()
    if not origin_coords:
        raise ValueError(f"Origin airport {flight_record['origin']} not found.")
    origin_lat, origin_lon = origin_coords
    
    # Get destination coordinates.
    cursor.execute("SELECT lat, lon FROM airports WHERE faa = ?", (flight_record['dest'],))
    dest_coords = cursor.fetchone()
    if not dest_coords:
        raise ValueError(f"Destination airport {flight_record['dest']} not found.")
    dest_lat, dest_lon = dest_coords
    
    # Compute flight bearing.
    flight_direction = calculate_bearing_vectorized(origin_lat, origin_lon, dest_lat, dest_lon)
    
    # Convert dep_time to a four-digit string.
    dep_time_str = str(flight_record['dep_time']).zfill(4)
    # Extract hour and minute.
    dep_hour_raw = int(dep_time_str[:-2])
    dep_minute = int(dep_time_str[-2:])
    # Round to nearest hour (if minutes >= 30, round up).
    rounded_dep_hour = dep_hour_raw + 1 if dep_minute >= 30 else dep_hour_raw
    if rounded_dep_hour == 24:
        rounded_dep_hour = 0
    
    # Retrieve weather data using the rounded departure hour.
    weather_query = """
    SELECT wind_speed, wind_dir
    FROM weather
    WHERE origin = ? AND year = ? AND month = ? AND day = ? AND hour = ?
    LIMIT 1
    """
    cursor.execute(weather_query, (flight_record['origin'],
                                   flight_record['year'],
                                   flight_record['month'],
                                   flight_record['day'],
                                   rounded_dep_hour))
    weather = cursor.fetchone()
    if not weather:
        raise ValueError("No matching weather record found for this flight's departure details.")
    wind_speed, wind_dir = weather
    
    # Compute inner product.
    inner_product = wind_speed * math.cos(math.radians(flight_direction - wind_dir))
    
    return {
        "flight": flight_record['flight'],
        "origin": flight_record['origin'],
        "dest": flight_record['dest'],
        "flight_direction": flight_direction,
        "dep_hour": rounded_dep_hour,
        "wind_speed": wind_speed,
        "wind_dir": wind_dir,
        "inner_product": inner_product
    }

def get_flight_records(limit=5_000):
    cursor = data_class.cursor
    query = "SELECT * FROM flights ORDER BY RANDOM() LIMIT ?"
    cursor.execute(query, (limit,))
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    records = [dict(zip(columns, row)) for row in rows]
    return records

def analyze_inner_product_vs_air_time():
    flights = get_flight_records()
    results = []
    for flight in flights:
        try:
            # Compute inner product for each flight.
            res = flight_wind_inner_product(flight)
            # Include air_time from the flight record.
            res['air_time'] = flight.get('air_time')
            results.append(res)
        except Exception as e:
            # Skip flights with missing weather or coordinate data.
            continue
    
    df = pd.DataFrame(results)
    if df.empty:
        return df, None  # Return the DataFrame and None for the figure
    
    # Define a function to assign sign labels.
    def ip_sign(x):
        if x > 0:
            return 'positive'
        elif x < 0:
            return 'negative'
        else:
            return 'zero'
    
    df['ip_sign'] = df['inner_product'].apply(ip_sign)
    
    # Create a Plotly scatter plot
    fig = px.scatter(
        df,
        x='inner_product',
        y='air_time',
        title="Scatter Plot: Inner Product vs Air Time",
        labels={'inner_product': 'Inner Product', 'air_time': 'Air Time (minutes)'},
        opacity=0.5
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis_title="Inner Product",
        yaxis_title="Air Time (minutes)",
        showlegend=False,
        height=600,
        width=1000
    )

    return df, fig