import math
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

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
def wind_direction():
    # Compute the direction the plane follows when flying to each airport from New York
    return
"""Write a function that computes the inner product between the flight direction
and the wind speed of a given flight."""
def inner_product(flight_id):
    # Compute the inner product between the flight direction and the wind speed
    return
"""Is there a relation between the sign of this inner product and the air time?"""
def relation():
    # Investigate the relation between the sign of the inner product and the air time
    return