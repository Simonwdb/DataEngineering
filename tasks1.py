import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('Data/airports.csv')

def plot_airports_map():
    """
    Plots a world map with points indicating the airports.
    """
    fig = px.scatter_geo(df, lat="lat", lon="lon",  color="alt", projection="natural earth")
    return fig.show()

def plot_airports_by_region(airports):
    """
    Identifies airports outside of the US and creates a separate US-only map.
    Also color codes airports by altitude.
    """
    pass

def is_american_faa(faa_code):
    """
    Not in the pdf but helper function to check if the airport is in America
    """
    # Filter the DataFrame for the given FAA code
    airport_row = df[df['faa'] == faa_code]

    if airport_row.empty:
        print(f"FAA code {faa_code} not found in the dataset.")
        return False

    # Get the timezone of the airport
    airport_tz = airport_row.iloc[0]['tzone']

    # Check if the timezone starts with "America/"
    return airport_tz.startswith("America/")

def plot_flight_from_nyc(faa_codes):
    """
    Plots a world map with a line from NYC (JFK) to the specified airport.
    """
    # Convert a single string to a list, if necessary
    if isinstance(faa_codes, str):
        faa_codes = [faa_codes]

    # NYC coordinates
    nyc_lat, nyc_lon = 40.7128, -74.0060

    # Filter the DataFrame for the selected airports (assuming your DataFrame is named `df`)
    df_selected = df[df["faa"].isin(faa_codes)]

    if df_selected.empty:
        print("No matching FAA codes found.")
        return

    # Check if all airports are in the US (requires a function is_american_faa)
    all_in_us = all(is_american_faa(faa) for faa in faa_codes)

    # Choose the projection type based on whether the airports are in the US
    projection_type = "albers usa" if all_in_us else "natural earth"

    # Create the base scatter_geo plot for the airports
    fig = px.scatter_geo(
        df_selected,
        lat="lat",
        lon="lon",
        hover_name="name",
        projection=projection_type,
        title="Routes from NYC to Selected Airport(s)"
    )

    # Add NYC point
    fig.add_trace(go.Scattergeo(
        lon=[nyc_lon],
        lat=[nyc_lat],
        mode='markers',
        marker=dict(size=8, color='red'),
        showlegend=False
    ))

    # Add lines from NYC to each airport
    for _, row in df_selected.iterrows():
        fig.add_trace(go.Scattergeo(
            lon=[nyc_lon, row['lon']],
            lat=[nyc_lat, row['lat']],
            mode='lines',
            line=dict(width=2, color='green', dash='dash'),
            showlegend=False
        ))

    # Update layout: restrict scope to USA if all airports are in the US
    if all_in_us:
        fig.update_geos(
            scope="usa",
            projection_type="albers usa"
        )
    else:
        fig.update_geos(
            scope="world"
        )

    return fig.show()

def compute_euclidean_distances(airports):
    """
    Computes the Euclidean distance from JFK to each airport and visualizes the distribution.
    """
    pass

def compute_geodesic_distances(airports):
    """
    Computes the geodesic distance from JFK to each airport using Earth's radius.
    """
    pass

def plot_timezones(airports):
    """
    Analyzes different time zones and represents the relative number of flights to them.
    """
    # test
    pass
