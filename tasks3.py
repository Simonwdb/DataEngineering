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

def verify_distances():
    # Compare the disatance from Pim to the distance in db
    return

"""For each flight, the origin from which it leaves can be found in the variable
origin in the table flights. Identify all diﬀerent airports in NYC from
which flights depart and save a DataFrame contain the information about those
airports from airports"""
def airports_in_nyc():
    # Get the airports in NYC from the flights table
    # Get the information about those airports from the airports table
    # Save the information in a DataFrame
    return
"""Write a function that takes a month and day and an airport in NYC as input,
and produces a figure similar to the one from part 1 containing all destinations
of flights on that day."""
def plot_destinations(month, day, airport):
    # Get the destinations of flights on that day
    # Plot the destinations
    return
"""Also write a function that returns statistics for that day, i.e. how many flights,
how many unique destinations, which destination is visited most often, etc."""
def get_statistics(month, day, airport):
    # Get the destinations of flights on that day
    # Get the statistics
    return
"""Write a function that, given a departing airport and an arriving airport, returns
a dict describing how many times each plane type was used for that flight
trajectory. For this task you will need to match the columns tailnum to type
in the table planes and match this to the tailnum s in the table flights."""
def plane_types(origin, destination):
    # Get the plane types for the flight trajectory
    # Return the dict
    return
"""Compute the average departure delay per flight for each of the airlines. Visualize
the results in a barplot with the full (rotated) names of the airlines on the x-axis."""
def average_delay_per_airline():
    # Compute the average departure delay per flight for each airline
    # Visualize the results in a barplot
    return
"""Write a function that takes as input a range of months and a destination and
returns the amount of delayed flights to that destination."""
def delayed_flights(months, destination):
    # Get the amount of delayed flights to the destination
    # Return the amount
    return

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