from utilities import *

data_class = Data()
import math
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from collections import defaultdict

conn = sqlite3.connect(r"Data/flights_database.db")

def test():
    print("Testing the functions from the tasks3.py file")

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

def airports_in_nyc(flights_df, airports_df):
    """
    Identifies all different NYC airports from which flights depart by:
      - Extracting the unique origin codes from the flights DataFrame (flights_df)
      - Filtering the airports DataFrame (airports_df) for those codes (using column 'faa')
    
    Returns:
        pd.DataFrame: A DataFrame containing the airport information for the NYC origin airports.
    """
    # Get the unique origin airport codes from the flights table
    origin_codes = flights_df['origin'].unique()
    print("Origin airport codes:", origin_codes)
    
    # Filter the airports DataFrame to include only airports present in the flights origins
    nyc_airports_df = airports_df[airports_df['faa'].isin(origin_codes)].copy()
    
    return nyc_airports_df

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
                         title=f"Destinations from {origin_airport} on {month}/{day}",
                         projection="natural earth")
    
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
def get_plane_types(cursor, origin, destination):
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

    cursor.execute(query_type_counts, (origin, destination))
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=[x[0] for x in cursor.description])
    result_dict = dict(zip(df['type'], df['count']))

    return result_dict

def get_plane_model_counts(cursor, origin, destination):
    query_model_counts = '''
    SELECT p.manufacturer || ' ' || p.model AS plane_model, COUNT(*) as count
    FROM flights AS f
    JOIN planes AS p ON f.tailnum = p.tailnum
    WHERE f.origin = ? AND f.dest = ?
    GROUP BY plane_model
    ORDER BY count DESC
    '''
    
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
def get_delayed_flights(flights_df, months, destination):
    # Get the amount of delayed flights to the destination
    # Return the amount
    delayed_flights = flights_df[
        flights_df['month'].isin(months) & 
        (flights_df['dest'] == destination) & 
        (flights_df['arr_delay'] > 0)
    ]
    return len(delayed_flights)

"""Write a function that takes a destination airport as input and returns the top 5
airplane manufacturers with planes departing to this destination. For this task,
you have to combine data from flights and planes."""
def top_manufacturers(conn, destination):
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
    df = pd.read_sql_query(query, conn, params=[destination])
    return df

"""Investigate whether there is a relationship between the distance of a flight and
the arrival delay time."""
def distance_vs_delay(conn):
    # Investigate the relationship between the distance of a flight and the arrival delay time
    querry = """select distance, arr_delay from flights where arr_delay is not null"""
    df = pd.read_sql_query(querry, conn)

    correlation = df['distance'].corr(df['arr_delay'])

    plt.figure(figsize=(8, 6))
    plt.scatter(df['distance'], df['arr_delay'], alpha=0.5)
    plt.xlabel("Distance (km)")
    plt.ylabel("Arrival Delay (minutes)")
    plt.title(f"Distance vs. Arrival Delay (corr = {correlation:.2f})")
    plt.show()
    #further inspection nessessary

    return correlation


"""Group the table flights by plane model using the tailnum. For each model,
compute the average speed by taking the average of the distance divided by flight
time over all flights of that model. Use this information to fill the column speed
in the table planes."""
import sqlite3


def fill_speed(conn):
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

    cursor = conn.cursor()
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
    print("Updated plane speeds successfully.")

    return

"""The wind direction is given in weather in degrees. Compute for each airport
the direction the plane follows when flying there from New York."""

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the avergage from point (lat1, lon1)
    to point (lat2, lon2) using the formula:
    
    θ = atan2( sin(Δlon)*cos(lat2),
               cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(Δlon) )
               
    The result is converted from radians to degrees and normalized to [0, 360).
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    delta_lon = lon2 - lon1
    
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    
    initial_bearing = math.atan2(x, y)
    # Convert bearing from radians to degrees and normalize to 0-360
    bearing_degrees = (math.degrees(initial_bearing) + 360) % 360
    return bearing_degrees

def average_bearing(bearings):
    """
    Compute the average bearing from a list of bearings (in degrees)
    by converting each to a unit vector, averaging, and then computing the angle.
    This is the proper way to average circular quantities.
    """
    sin_sum = sum(math.sin(math.radians(b)) for b in bearings)
    cos_sum = sum(math.cos(math.radians(b)) for b in bearings)
    avg_angle_rad = math.atan2(sin_sum, cos_sum)
    avg_angle_deg = (math.degrees(avg_angle_rad) + 360) % 360
    return avg_angle_deg

cursor = conn.cursor()
query = """
SELECT f.origin, f.dest, o.lat as origin_lat, o.lon as origin_lon, d.lat as dest_lat, d.lon as dest_lon
FROM flights f
JOIN airports o ON f.origin = o.faa
JOIN airports d ON f.dest = d.faa;
"""
cursor.execute(query)
rows = cursor.fetchall()

dest_bearings = defaultdict(list)

for origin, dest, origin_lat, origin_lon, dest_lat, dest_lon in rows:
    bearing = calculate_bearing(origin_lat, origin_lon, dest_lat, dest_lon)
    dest_bearings[dest].append(bearing)

# Compute the average bearing for each destination airport.
# The average is computed properly using vector averaging.
average_bearings = {dest: average_bearing(bearings) for dest, bearings in dest_bearings.items()}

cursor.execute("SELECT faa, name FROM airports")
airport_names = dict(cursor.fetchall())

print("Average Bearing from all NYC departure airports to each destination airport:")
for dest, avg_bearing in average_bearings.items():
    name = airport_names.get(dest, "Unknown")
    print(f"Airport: {dest} - {name:40s} | Average Bearing: {avg_bearing:.2f}°")


"""Write a function that computes the inner product between the flight direction
and the wind speed of a given flight."""

INDEX = 5

def flight_wind_inner_product(flight_record):
    cursor = conn.cursor()
    
    # Retrieve coordinates for the origin airport.
    cursor.execute("SELECT lat, lon FROM airports WHERE faa = ?", (flight_record['origin'],))
    origin_coords = cursor.fetchone()
    if not origin_coords:
        raise ValueError(f"Origin airport {flight_record['origin']} not found.")
    origin_lat, origin_lon = origin_coords
    
    # Retrieve coordinates for the destination airport.
    cursor.execute("SELECT lat, lon FROM airports WHERE faa = ?", (flight_record['dest'],))
    dest_coords = cursor.fetchone()
    if not dest_coords:
        raise ValueError(f"Destination airport {flight_record['dest']} not found.")
    dest_lat, dest_lon = dest_coords
    
    # Compute the flight's bearing (direction) from origin to destination.
    flight_direction = calculate_bearing(origin_lat, origin_lon, dest_lat, dest_lon)
    
    dep_time_str = str(flight_record['dep_time']).zfill(4)
    dep_hour_raw = int(dep_time_str[:-2])
    dep_minute = int(dep_time_str[-2:])
    
    rounded_dep_hour = dep_hour_raw + 1 if dep_minute >= 30 else dep_hour_raw
 
    if rounded_dep_hour == 24:
        rounded_dep_hour = 0
    
    # Retrieve the corresponding weather data using the rounded departure hour.
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
    
    # Compute the inner product: wind_speed * cos( flight_direction - wind_dir )
    inner_product = wind_speed * math.cos(math.radians(flight_direction - wind_dir))
    
    result = {
        "flight": flight_record['flight'],
        "origin": flight_record['origin'],
        "dest": flight_record['dest'],
        "flight_direction": flight_direction,
        "dep_hour": rounded_dep_hour,
        "wind_speed": wind_speed,
        "wind_dir": wind_dir,
        "inner_product": inner_product
    }
    return result

def get_flight_record_by_index(index):
    cursor = conn.cursor()
    query = "SELECT * FROM flights LIMIT 1 OFFSET ?"
    cursor.execute(query, (index,))
    row = cursor.fetchone()
    if row is None:
        raise ValueError(f"No flight found at position {index}.")
    columns = [desc[0] for desc in cursor.description]
    flight_record = dict(zip(columns, row))
    return flight_record

if __name__ == "__main__":   
    try:
        # Retrieve a flight record by its position (e.g., the 2nd flight; 0-based index).
        flight_record = get_flight_record_by_index(INDEX)
        print("Selected Flight Record:")
        for key, value in flight_record.items():
            print(f"{key}: {value}")
        
        print("\nComputing wind inner product for the selected flight...")
        result = flight_wind_inner_product(flight_record)
        print("\nFlight Wind Inner Product Details:")
        for key, value in result.items():
            print(f"{key}: {value}")
    except Exception as e:
        print("Error: Index out of range", e)

"""Is there a relation between the sign of this inner product and the air time?"""

def flight_wind_inner_product(flight_record):
    cursor = conn.cursor()
    
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
    flight_direction = calculate_bearing(origin_lat, origin_lon, dest_lat, dest_lon)
    
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

def get_flight_records(limit=10000):
    cursor = conn.cursor()
    query = "SELECT * FROM flights LIMIT ?"
    cursor.execute(query, (limit,))
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    records = [dict(zip(columns, row)) for row in rows]
    return records

def analyze_inner_product_vs_air_time(sample_size=10000):
    flights = get_flight_records(sample_size)
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
        print("No valid flight records found for analysis.")
        return df
    
    # Define a function to assign sign labels.
    def ip_sign(x):
        if x > 0:
            return 'positive'
        elif x < 0:
            return 'negative'
        else:
            return 'zero'
    
    df['ip_sign'] = df['inner_product'].apply(ip_sign)
    
    # Summary: average air_time by inner product sign.
    summary = df.groupby('ip_sign')['air_time'].mean()
    print("Average air_time by inner product sign:")
    print(summary)
       
    # Scatter plot: inner product vs air_time.
    plt.figure(figsize=(8,6))
    plt.scatter(df['inner_product'], df['air_time'], alpha=0.5)
    plt.xlabel("Inner Product")
    plt.ylabel("Air Time (minutes)")
    plt.title("Scatter Plot: Inner Product vs Air Time")
    plt.show()
    
    return df

if __name__ == "__main__":
    try:
        df_analysis = analyze_inner_product_vs_air_time(sample_size=10000)
    except Exception as e:
        print("Error during analysis", e)