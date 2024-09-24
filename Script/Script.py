## MANAGING THE TRANSITION TO ELECTRIC VEHICLES ##


### CLUSTURS

## Install Package

pip install geopy
pip install folium

## Reading Data from Excel

# import libraries
import pandas as pd
from geopy.distance import geodesic
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster

# Load the Excel file
file_path = "Inputs.xlsx"

# Read the stations and districts data
stations = pd.read_excel(file_path, sheet_name='Petrol_Stations')
districts = pd.read_excel(file_path, sheet_name='Postcodes')

# Display loaded data to confirm
print("Stations Data:")
print(stations.head())
print("\nDistricts Data:")
print(districts.head())

## Calculate Distance Matrix between Stations and Districts

# Initialize an empty distance matrix for stations-to-districts
station_to_district_matrix = pd.DataFrame(index=stations['Station_ID'], columns=districts['Postcode'])

# Calculate distances from each station to each district
for i, station in stations.iterrows():
    for j, district in districts.iterrows():
        coord_station = (station['Station_Latitude'], station['Station_Longitude'])
        coord_district = (district['District_Latitude'], district['District_Longitude'])
        distance = geodesic(coord_station, coord_district).km
        station_to_district_matrix.loc[station['Station_ID'], district['Postcode']] = distance

# Print the Station-to-District Distance Matrix
print("Station-to-District Distance Matrix:")
print(station_to_district_matrix)

## Calculate Distance Matrix between All Stations

# Initialize an empty distance matrix for stations-to-stations
station_to_station_matrix = pd.DataFrame(index=stations['Station_ID'], columns=stations['Station_ID'])

# Calculate distances between each pair of stations
for i, station1 in stations.iterrows():
    for j, station2 in stations.iterrows():
        if i != j:  # No need to calculate distance from a station to itself
            coord_station1 = (station1['Station_Latitude'], station1['Station_Longitude'])
            coord_station2 = (station2['Station_Latitude'], station2['Station_Longitude'])
            distance = geodesic(coord_station1, coord_station2).km
            station_to_station_matrix.loc[station1['Station_ID'], station2['Station_ID']] = distance
        else:
            station_to_station_matrix.loc[station1['Station_ID'], station2['Station_ID']] = 0

# Print the Station-to-Station Distance Matrix
print("Station-to-Station Distance Matrix:")
print(station_to_station_matrix)

## Normalize the Distance Matrices

# Normalize the station-to-district matrix
scaler = StandardScaler()
normalized_station_to_district = scaler.fit_transform(station_to_district_matrix.fillna(0))

# Normalize the station-to-station matrix
normalized_station_to_station = scaler.fit_transform(station_to_station_matrix.fillna(0))

# Convert back to DataFrames
normalized_station_to_district_df = pd.DataFrame(normalized_station_to_district, index=station_to_district_matrix.index, columns=station_to_district_matrix.columns)
normalized_station_to_station_df = pd.DataFrame(normalized_station_to_station, index=station_to_station_matrix.index, columns=station_to_station_matrix.columns)

print("Normalized Station-to-District Distance Matrix:")
print(normalized_station_to_district_df)

print("Normalized Station-to-Station Distance Matrix:")
print(normalized_station_to_station_df)

## Elbow method to find the Number of Clusters

inertia = []
range_clusters = range(1, 11)  # Test for 1 to 10 clusters

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_station_to_district)
    inertia.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 6))
plt.plot(range_clusters, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

## K-Means Clustering

# Define the number of clusters
n_clusters = 3  # This can be set based on the elbow method result

# Apply K-Means Clustering on normalized station-to-district matrix
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
station_to_district_matrix['Cluster'] = kmeans.fit_predict(normalized_station_to_district)

# Visualize the Clusters
plt.figure(figsize=(8, 6))
plt.scatter(range(len(station_to_district_matrix)), station_to_district_matrix['Cluster'], c=station_to_district_matrix['Cluster'], cmap='viridis')
plt.xlabel('Petrol Stations')
plt.ylabel('Cluster')
plt.title('K-Means Clustering of Petrol Stations Based on Distance to Districts')
plt.show()

print("K-Means Clustering Results for Stations-to-Districts:")
print(station_to_district_matrix[['Cluster']])

## Hierarchical Clustering

# Perform hierarchical clustering using the Ward method on station-to-station distance matrix
Z = linkage(normalized_station_to_station, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, labels=station_to_station_matrix.index)
plt.title('Hierarchical Clustering Dendrogram for Petrol Stations')
plt.xlabel('Petrol Stations')
plt.ylabel('Distance')
plt.show()

# Print the structure of the DataFrames
print("station_to_district_matrix structure:")
print(station_to_district_matrix.info())
print("\nstations structure:")
print(stations.info())

# Print a few rows of the Cluster column
print("\nFirst few rows of the Cluster column:")
print(station_to_district_matrix['Cluster'].head())

# Check if all station IDs in stations DataFrame are in station_to_district_matrix
missing_stations = set(stations['Station_ID']) - set(station_to_district_matrix.index)
print("\nMissing stations:", missing_stations)

# Print unique values in the Cluster column
print("\nUnique values in Cluster column:")
print(station_to_district_matrix['Cluster'].unique())

## Visualize Clusters on Map with Folium

# Initialize the map centered around Southampton
m = folium.Map(location=[50.9097, -1.4044], zoom_start=8)

# Create a marker cluster
marker_cluster = MarkerCluster().add_to(m)

# Add markers for each station with cluster information
for i, row in stations.iterrows():
    station_id = row['Station_ID']
    
    # Check if the station_id exists in the station_to_district_matrix
    if station_id in station_to_district_matrix.index:
        cluster_series = station_to_district_matrix.loc[station_id, 'Cluster']
        
        # Handle different types of cluster_series
        if isinstance(cluster_series, pd.Series):
            cluster_id = cluster_series.iloc[0]
        else:
            cluster_id = cluster_series
        
        # Determine color based on the cluster
        if cluster_id == 0:
            color = 'blue'
        elif cluster_id == 1:
            color = 'green'
        elif cluster_id == 2:
            color = 'red'
        else:
            color = 'gray'  # Default color for any unexpected cluster values
    else:
        print(f"Warning: Station ID {station_id} not found in cluster data")
        color = 'black'  # Color for stations without cluster data
    
    # Add a marker for each station
    folium.Marker(
        location=[row['Station_Latitude'], row['Station_Longitude']],
        popup=f"Station ID: {station_id} - Cluster: {cluster_id if 'cluster_id' in locals() else 'N/A'}",
        icon=folium.Icon(color=color)
    ).add_to(marker_cluster)

# Save map to an HTML file
m.save('clusters_map.html')
print("Map with clustered petrol stations has been saved to 'clusters_map.html'.")

# Save all outputs to Excel file
output_path = "Distance_Matrix_Outputs.xlsx"
with pd.ExcelWriter(output_path) as writer:
    station_to_district_matrix.to_excel(writer, sheet_name='Station_to_District_Distances')
    station_to_station_matrix.to_excel(writer, sheet_name='Station_to_Station_Distances')
    normalized_station_to_district_df.to_excel(writer, sheet_name='Normalized_Station_to_District')
    normalized_station_to_station_df.to_excel(writer, sheet_name='Normalized_Station_to_Station')
    station_to_district_matrix[['Cluster']].to_excel(writer, sheet_name='KMeans_Clustering_Results')

print(f"Outputs have been saved to {output_path}")



## OPTIMISATION MODEL USING LINEAR PROGRAMMING (PULP)

# Import libraries

from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import pandas as pd
import numpy as np

# Load the updated Excel file with clustering information
file_path = 'Inputs.xlsx'

# Read the data
petrol_stations = pd.read_excel(file_path, sheet_name='Petrol_Stations')
station_to_district_distance = pd.read_excel(file_path, sheet_name='Station_to_District_Distances')

# Ensure 'Station_ID' is correctly identified and no NaN values are present
station_to_district_distance.dropna(subset=['Station_ID', 'Postcode'], inplace=True)

# Extract unique values for use in the model
station_ids = list(set(petrol_stations['Station_ID'].unique()).intersection(station_to_district_distance['Station_ID'].unique()))
postcodes = station_to_district_distance['Postcode'].unique()

# Read clustering information
cluster_ids = petrol_stations['Cluster_ID'].unique() 

# Set a specific value for x (distance threshold)
x_value = 1.5  # Fixed at 1.5 km for the specified output

# Create the optimization problem
model = LpProblem("Maximize_Coverage", LpMaximize)

# Decision variables for whether a station is open (1) or closed (0)
y = {station: LpVariable(f'y_{station}', cat='Binary') for station in station_ids}

# Decision variables for whether a postcode is covered (1) or not (0)
z = {postcode: LpVariable(f'z_{postcode}', cat='Binary') for postcode in postcodes}

# Objective: Maximize the number of covered postcodes
model += lpSum(z[postcode] for postcode in z), "Total_Covered_Postcodes"

# Coverage Constraint: Ensure each postcode is covered by at least one open station within x_value
for postcode in postcodes:
    model += lpSum(
        y[station_id] for station_id in station_ids 
        if not station_to_district_distance[
            (station_to_district_distance['Station_ID'] == station_id) & 
            (station_to_district_distance['Postcode'] == postcode) &
            (station_to_district_distance['Distance (km)'] <= x_value)
        ].empty
    ) >= z[postcode], f"Coverage_Constraint_{postcode}"

# Station Limitation Constraint: Maximum 40% of stations should remain open (60% will close)
max_open_stations = int(0.4 * len(station_ids))
model += lpSum(y[station] for station in y) <= max_open_stations, "Max_Open_Stations"

# Clustering Constraint: Ensure at least one station remains open in each cluster
for cluster in cluster_ids:
    stations_in_cluster = petrol_stations[petrol_stations['Cluster_ID'] == cluster]['Station_ID'].unique()
    model += lpSum(y[station] for station in stations_in_cluster) >= 1, f"Cluster_Constraint_{cluster}"

# Solve the model
model.solve()

# Collect results
results_output = []
district_postcodes_covered = set()

for station in station_ids:
    open_close_status = "Open" if y[station].value() == 1 else "Close"
    covered_districts = station_to_district_distance[
        (station_to_district_distance['Station_ID'] == station) &
        (station_to_district_distance['Distance (km)'] <= x_value)
    ]['Postcode'].unique()
    num_covered_districts = len(covered_districts)

    results_output.append({
        'Station ID': station,
        'Name': petrol_stations.loc[petrol_stations['Station_ID'] == station, 'Station_Name'].values[0],
        'Districts Covered': num_covered_districts,
        'Status': open_close_status
    })

    if open_close_status == "Open":
        district_postcodes_covered.update(covered_districts)

# Convert results to a DataFrame for better display
results_df = pd.DataFrame(results_output)

# Print the results in column format
print("Results for x = 1.5 km:")
print(results_df.to_string(index=False))

# Calculate the overall number of covered districts and optimal solution value
num_covered_districts_total = len(district_postcodes_covered)
optimal_solution_percentage = (num_covered_districts_total / len(postcodes)) * 100

# Print the overall results
print(f"\nTotal number of districts covered: {num_covered_districts_total}")
print(f"Optimal solution value: {optimal_solution_percentage:.2f}%")


