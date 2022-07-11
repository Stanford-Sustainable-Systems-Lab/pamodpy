import requests
import pandas as pd
import geopandas as gpd
import os
import pickle
from pamodpy.Station import Station
from pamodpy.EVSE import EVSE

# Fetch Data from AFDC
# https://afdc.energy.gov/fuels/electricity_locations.html#/find/nearest?fuel=ELEC
api_key = 'nREpbKS1vtjm8B4Tle8ZzFJ5LaKbSH6tWmTXcwKk'
response = requests.get('https://developer.nrel.gov/api/alt-fuel-stations/v1.csv',
                        params={'api_key': api_key,
                                'fuel_type': 'ELEC',
                                'state': 'CA'})

with open('SF_charging_stations.csv', 'w') as f:
    f.write(response.text)

# Load AFDC Data as DataFrame and convert to GeoDataFrame
df = pd.read_csv('SF_charging_stations.csv', encoding='ISO-8859-1')
df = df[['Station Name', 'Street Address', 'City', 'State', 'ZIP', 'EV Level1 EVSE Num', 'EV Level2 EVSE Num',
         'EV DC Fast Count', 'EV Other Info', 'EV Network', 'EV Network Web', 'Latitude', 'Longitude',
         'Date Last Confirmed', 'ID', 'Updated At', 'Owner Type Code', 'EV Connector Types']]
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))

# Load TAZ GeoDataFrame
SF_map = gpd.read_file(os.path.join("SF_190",
                     'Justin-Luke---Academic_SF-TAZ-with-added-boundary-pass-through_ZoneSet',
                     'zone_set_SF_TAZ_with_added_boundary_pass_through.shp'))
SF_map = SF_map.set_index('name')
SF_map.index = SF_map.index.astype(int)
SF_map = SF_map.sort_index()

# Merged GeoDataFrame; column "index_right" is the TAZ corresponding to the station (each row)
joined_gdf = gpd.sjoin(gdf, SF_map, op='within')
print(joined_gdf.size)
print(joined_gdf.columns)

charge_stations_190 = {}
for l in range(1, 191):
    charge_stations_190[l] = []
    UMax_charge_l_level2 = joined_gdf[joined_gdf.index_right == l]['EV Level2 EVSE Num'].sum()
    UMax_charge_l_dc = joined_gdf[joined_gdf.index_right == l]['EV DC Fast Count'].sum()
    station = Station(l, l)
    if UMax_charge_l_level2 != 0:
        station.EVSEs.append(EVSE('AC 7.7kW', UMax_charge_l_level2, station))
    if UMax_charge_l_dc != 0:
        station.EVSEs.append(EVSE('DC 50kW', UMax_charge_l_dc, station))
    if station.EVSEs != []:
        charge_stations_190[l].append(station)

cluster_to_taz = {
            1: [56, 57],
            2: [52, 62, 65, 66],
            3: [43, 48, 49, 50, 51, 70, 71, 72],
            4: [6, 7, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 26, 39, 40, 41, 44, 45, 46, 47, 73, 74, 75],
            5: [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26, 37, 38, 42],
            6: [54, 55, 58, 59, 60],
            7: [53, 61, 63, 64, 67, 90],
            8: [68, 69, 83, 84, 85, 86, 87, 88, 89],
            9: [8, 9, 10, 11, 20, 76, 77, 78, 79, 80, 81, 82, 104, 105, 106, 107],
            10: [18, 19, 21, 108, 109, 110],
            11: [178, 179, 180, 181, 184],
            12: [172, 173, 174, 175, 176, 177, 185],
            13: [91, 92, 93, 94, 95, 96, 129, 171],
            14: [97, 98, 99, 100, 101, 102, 103, 116, 117, 118, 119, 122, 123, 128],
            15: [111, 112, 113, 114, 115, 120, 121, 142],
            16: [182, 183, 186, 187],
            17: [169, 170, 188],
            18: [130, 131, 132, 133, 134],
            19: [124, 125, 126, 127, 135, 136, 137, 138, 152],
            20: [139, 140, 141, 143, 144, 145, 146, 147, 150],
            21: [190],
            22: [168, 189],
            23: [161, 162, 163, 164, 165, 166, 167],
            24: [155, 156, 157, 158, 159, 160],
            25: [148, 149, 151, 153, 154]
}

charge_stations_25 = {}
for l in range(1, 26):
    charge_stations_25[l] = []
    UMax_charge_l_level2 = joined_gdf[joined_gdf.index_right.isin(cluster_to_taz[l])]['EV Level2 EVSE Num'].sum()
    UMax_charge_l_dc = joined_gdf[joined_gdf.index_right.isin(cluster_to_taz[l])]['EV DC Fast Count'].sum()
    station = Station(l, l)
    if UMax_charge_l_level2 != 0:
        station.EVSEs.append(EVSE('AC 7.7kW', UMax_charge_l_level2, station))
    if UMax_charge_l_dc != 0:
        station.EVSEs.append(EVSE('DC 50kW', UMax_charge_l_dc, station))
    if station.EVSEs != []:
        charge_stations_25[l].append(station)


print(charge_stations_190)
print(charge_stations_25)

with open('SF_190/SF_charging_stations_to_190_TAZ.p', 'wb') as f:
    pickle.dump(charge_stations_190, f)

with open('SF_25/SF_charging_stations_to_25_cluster.p', 'wb') as f:
    pickle.dump(charge_stations_25, f)
