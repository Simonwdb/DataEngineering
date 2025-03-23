import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def plot_airports_map(airports_df):
    """
    Plots a world map with points indicating the airports.
    """
    fig = px.scatter_geo(airports_df, lat="lat", lon="lon",  color="alt", projection="natural earth")
    return fig.show()

def plot_airports_by_region(airports_df):
    """
    Identifies airports outside of the US and creates a separate US-only map.
    Also color codes airports by altitude.
    """
    america_df = airports_df.copy(deep=True)
    america_df['tzone'].fillna('Unknown', inplace=True)
    america_df = america_df[america_df['tzone'].str.startswith('America')]
    fig = px.scatter_geo(
        america_df,
        lat='lat',
        lon='lon',
        hover_name='name',
        scope='usa',
        color='alt'
    )
    return fig
    

def is_american_faa(faa_code, airports_df):
    """
    Not in the pdf but helper function to check if the airport is in America
    """
    # Filter the DataFrame for the given FAA code
    airport_row = airports_df[airports_df['faa'] == faa_code]

    if airport_row.empty:
        print(f"FAA code {faa_code} not found in the dataset.")
        return False

    # Get the timezone of the airport
    airport_tz = airport_row.iloc[0]['tzone']

    # Check if the timezone starts with "America/"
    return airport_tz.startswith("America/")

def plot_flight_from_nyc(faa_codes, airports_df):
    """
    Plots a world map with a line from NYC (JFK) to the specified airport.
    """
    # Convert a single string to a list, if necessary
    if isinstance(faa_codes, str):
        faa_codes = [faa_codes]

    # NYC coordinates
    nyc_lat, nyc_lon = 40.7128, -74.0060

    # Filter the DataFrame for the selected airports (assuming your DataFrame is named `df`)
    df_selected = airports_df[airports_df["faa"].isin(faa_codes)]

    if df_selected.empty:
        return None

    # Check if all airports are in the US (requires a function is_american_faa)
    all_in_us = all(is_american_faa(faa, airports_df) for faa in faa_codes)

    # Choose the projection type based on whether the airports are in the US
    projection_type = "albers usa" if all_in_us else "natural earth"

    # Create the base scatter_geo plot for the airports
    fig = px.scatter_geo(
        df_selected,
        lat="lat",
        lon="lon",
        hover_name="name",
        projection=projection_type,
        title= f"Route from NYC to Selected Airport: {airports_df.loc[airports_df['faa'] == faa_codes[0], 'name'].values[0]}"
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

    return fig


def compute_euclidean_distances(airports_df):
    """
    Computes the Euclidean distance from JFK to each airport in the provided DataFrame,
    visualizes the distribution as a histogram, and returns a new DataFrame with the computed distances.
    
    Parameters:
        airports (pd.DataFrame): DataFrame containing airport information with at least the columns:
                                 'name', 'lat', 'lon'
    
    Returns:
        pd.DataFrame: A copy of the input DataFrame containing two new columns:
                      'euclidean_distance_deg' (distance in degrees)
                      'euclidean_distance_km' (distance in kilometers, approximated as degrees * 111)
                      Only rows with a realistic distance (<= 180°) are returned.
    """
    # Zoek naar de rij voor "John F Kennedy International Airport"
    jfk_row = airports_df[airports_df['name'].str.contains("John F Kennedy International Airport", 
                                                       case=False, na=False)]
    if jfk_row.empty:
        raise ValueError("JFK airport not found in the dataset!")
    else:
        jfk = jfk_row.iloc[0]
        jfk_lat_deg = jfk['lat']  # graden
        jfk_lon_deg = jfk['lon']  # graden
        print(f"JFK Position: lat = {jfk_lat_deg}, lon = {jfk_lon_deg}")
    
    # Werk met een kopie zodat de originele DataFrame niet wordt aangepast
    airports_copy = airports_df.copy()
    
    # Bereken de Euclidische afstand in graden vanaf JFK
    airports_copy['euclidean_distance_deg'] = np.sqrt(
        (airports_copy['lat'] - jfk_lat_deg)**2 + (airports_copy['lon'] - jfk_lon_deg)**2
    )
    
    # Filter: houd alleen realistische afstanden (<= 180 graden)
    filtered_airports = airports_copy[airports_copy['euclidean_distance_deg'] <= 180].copy()
    
    # Converteer de afstand van graden naar kilometers (benadering: 1 graad ≈ 111 km)
    filtered_airports['euclidean_distance_km'] = filtered_airports['euclidean_distance_deg'] * 111
    
    # Visualiseer de verdeling van de Euclidische afstanden (in km)
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_airports['euclidean_distance_km'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Euclidean Distance (km)')
    plt.ylabel('Number of Airports')
    plt.title('Distribution of Euclidean Distances from JFK (approx. km)')
    plt.grid(True)
    plt.xlim(left=0)
    plt.show()
    
    return filtered_airports



def compute_geodesic_distances(airports_df):
    """
    Computes the geodesic distance from JFK to each airport in the provided DataFrame,
    visualizes the distribution as a histogram, and returns a new DataFrame with the computed distances.
    
    Parameters:
        airports (pd.DataFrame): DataFrame containing airport information with at least the columns:
                                 'name', 'lat', 'lon'
    
    Returns:
        pd.DataFrame: A copy of the input DataFrame containing a new column:
                      'geodesic_distance_km' (geodesic distance in kilometers)
                      Only rows with realistic distances (<= 20000 km) are returned.
    """
    # Zoek naar JFK in de dataset
    jfk_row = airports_df[airports_df['name'].str.contains("John F Kennedy International Airport", case=False, na=False)]
    if jfk_row.empty:
        raise ValueError("JFK airport not found in the dataset!")
    else:
        jfk = jfk_row.iloc[0]
        jfk_lat_deg = jfk['lat']  # graden
        jfk_lon_deg = jfk['lon']  # graden
        print(f"JFK Position: lat = {jfk_lat_deg}, lon = {jfk_lon_deg}")
    
    # Maak een kopie zodat de originele DataFrame niet wordt aangepast
    airports_copy = airports_df.copy()
    
    # Aardstraal in kilometers
    R = 6371
    
    # Converteer de breedte- en lengtegraad van alle luchthavens en van JFK naar radialen
    lat_rad = np.radians(airports_copy['lat'])
    lon_rad = np.radians(airports_copy['lon'])
    jfk_lat_rad = np.radians(jfk_lat_deg)
    jfk_lon_rad = np.radians(jfk_lon_deg)
    
    # Bereken de verschillen (in radialen)
    dphi = lat_rad - jfk_lat_rad      # Δφ
    dlambda = lon_rad - jfk_lon_rad   # Δλ
    phi_m = (lat_rad + jfk_lat_rad) / 2  # Middenwaarde van de breedtegraad
    
    # Bereken de geodetische afstand volgens de formule:
    # d = R * sqrt((2*sin(Δφ/2)*cos(Δλ/2))^2 + (2*cos(φ_m)*sin(Δλ/2))^2)
    term1 = 2 * np.sin(dphi / 2) * np.cos(dlambda / 2)
    term2 = 2 * np.cos(phi_m) * np.sin(dlambda / 2)
    airports_copy['geodesic_distance_km'] = R * np.sqrt(term1**2 + term2**2)
    
    # Filter onrealistische waarden (bijvoorbeeld > 20000 km)
    filtered_airports = airports_copy[airports_copy['geodesic_distance_km'] <= 20000].copy()
    
    # Visualiseer de verdeling van de geodetische afstanden
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_airports['geodesic_distance_km'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Geodesic Distance (km)')
    plt.ylabel('Number of Airports')
    plt.title('Distribution of Geodesic Distances from JFK')
    plt.grid(True)
    plt.xlim(left=0)
    plt.show()
    
    return filtered_airports

def plot_timezones(airports_df):
    airports_df["timezone"] = pd.to_numeric(airports_df["tz"], errors='coerce')

    time_zone_counts = airports_df["timezone"].value_counts().reset_index()
    time_zone_counts.columns = ["Time Zone", "Number of Airports"]

    # Sort by count in descending order
    time_zone_counts = time_zone_counts.sort_values("Number of Airports", ascending=False)

    fig = px.bar(time_zone_counts, x="Time Zone", y="Number of Airports",
                 title="Distribution of Airports by Time Zone",
                 labels={"Time Zone": "Time Zone (UTC Offset)", "Number of Airports": "Count"},
                 text_auto=True,
                 color="Number of Airports", color_continuous_scale="Viridis")

    # Ensure the x-axis shows the correct time zone values
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=time_zone_counts["Time Zone"]))

    return fig

def plot_airport_altitude_distribution(airports_df):
    """
    Plots distribution of airport frequency vs its altitude
    """
    fig = px.histogram(airports_df, x="alt", nbins=50, 
                   title="Distribution of Airport Altitudes",
                   labels={"alt": "Altitude (ft)", "Airport frequency": "Number of Airports"})

    return fig
