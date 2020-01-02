import pandas as pd
import numpy as np
from geopy.distance import great_circle

W_1 = np.load('W_Relation_Matrix_1W_SGD_point_01.npy')
W_2_dis = np.load('W_Relation_Disaster_2W_SGD_point_01.npy')
W_2_reg = np.load('W_Relation_Regular_2W_SGD_point_01.npy')

dist = pd.read_excel('Node_Features_Weekly.xlsx', sheet_name = 'Node Distance')
dist = dist.reset_index()
dist = dist.rename(columns = {"level_0": "Destination ID", "level_1": "Origin ID",\
                            "Destination Cluster ID": "Miles"})
table = pd.pivot_table(dist, values = "Miles", index = ["Origin ID"], \
                       columns = ["Destination ID"], fill_value = 0)
D = table.to_numpy()
np.save('Data_Distance_Matrix', D)
Dt = D.transpose()

print(np.corrcoef(W_1.reshape(136*136), D.reshape(136*136)))
print(np.corrcoef(W_1.reshape(136*136), Dt.reshape(136*136)))

print(np.corrcoef(W_2_dis.reshape(136*136), D.reshape(136*136)))
print(np.corrcoef(W_2_dis.reshape(136*136), Dt.reshape(136*136)))

print(np.corrcoef(W_2_reg.reshape(136*136), D.reshape(136*136)))
print(np.corrcoef(W_2_reg.reshape(136*136), Dt.reshape(136*136)))

vol = pd.read_excel('Node_Features_Weekly.xlsx', sheet_name = 'Lane Volume')
vol = vol.reset_index()
vol = vol.rename(columns = {"level_0": "Destination ID", "level_1": "Origin ID",\
                            "Destination Cluster ID": "Volume"})
table2 = pd.pivot_table(vol, values = "Volume", index = ["Origin ID"], \
                       columns = ["Destination ID"], fill_value = 0)
V = table2.to_numpy()
Vt = V.transpose()

print(np.corrcoef(W_1.reshape(136*136), V.reshape(136*136)))
print(np.corrcoef(W_1.reshape(136*136), Vt.reshape(136*136)))

print(np.corrcoef(W_2_dis.reshape(136*136), V.reshape(136*136)))
print(np.corrcoef(W_2_dis.reshape(136*136), Vt.reshape(136*136)))

print(np.corrcoef(W_2_reg.reshape(136*136), V.reshape(136*136)))
print(np.corrcoef(W_2_reg.reshape(136*136), Vt.reshape(136*136)))

lat_long = pd.read_excel('Node_Features_Weekly.xlsx', sheet_name = 'Node Lat Long')
clusters = lat_long['Cluster ID'].unique()
Geo_D = np.zeros((136,136))
for i in range(136):
    o_lat = lat_long[lat_long['Cluster ID'] == clusters[i]].Latitude.values
    o_long = lat_long[lat_long['Cluster ID'] == clusters[i]].Longitude.values
    o = (o_lat, o_long)
    for j in range(136):
        d_lat = lat_long[lat_long['Cluster ID'] == clusters[j]].Latitude.values
        d_long = lat_long[lat_long['Cluster ID'] == clusters[j]].Longitude.values
        d = (d_lat, d_long)
        
        Geo_D[i,j] = great_circle(o, d).miles
np.save('Geo_Distance_Matrix', Geo_D)

print(np.corrcoef(W_1.reshape(136*136), Geo_D.reshape(136*136)))

print(np.corrcoef(W_2_dis.reshape(136*136), Geo_D.reshape(136*136)))

print(np.corrcoef(W_2_reg.reshape(136*136), Geo_D.reshape(136*136)))
