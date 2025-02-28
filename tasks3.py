import math
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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