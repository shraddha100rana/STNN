import pandas as pd
import numpy as np

dates_df = pd.read_excel('Node_Features_Daily.xlsx', sheet_name = 'Dates')
dates = dates_df.Date.unique()

lat_long = pd.read_excel('Node_Features_Daily.xlsx', sheet_name = 'Node Lat Long')
clusters = lat_long['Cluster ID'].unique()

OB_SH_NF = pd.read_excel('Node_Features_Daily.xlsx', sheet_name = 'OB_SH_NF')
OB_SH_F = pd.read_excel('Node_Features_Daily.xlsx', sheet_name = 'OB_SH_F')
OB_LH_NF = pd.read_excel('Node_Features_Daily.xlsx', sheet_name = 'OB_LH_NF')
OB_LH_F = pd.read_excel('Node_Features_Daily.xlsx', sheet_name = 'OB_LH_F')
IB_SH_NF = pd.read_excel('Node_Features_Daily.xlsx', sheet_name = 'IB_SH_NF')
IB_SH_F = pd.read_excel('Node_Features_Daily.xlsx', sheet_name = 'IB_SH_F')
IB_LH_NF = pd.read_excel('Node_Features_Daily.xlsx', sheet_name = 'IB_LH_NF')
IB_LH_F = pd.read_excel('Node_Features_Daily.xlsx', sheet_name = 'IB_LH_F')

FEMA = pd.read_excel('Node_Features_Daily.xlsx', sheet_name = 'FEMA Declaration')
cluster_state = pd.read_excel('Node_Features_Daily.xlsx', sheet_name = 'Cluster State')

#dimensions of X
T = len(dates)
n = len(clusters)
m = (2*2*2*4)+2+1
#dimension of Y
l = (2*2)

#Avg CPM, Max CPM, Vol, Lanes, Lat, Long
X = np.zeros((T,n,m)) 
#Avg CPM
Y = np.zeros((T,n,l))

for i in range(n):
    a = OB_SH_NF[OB_SH_NF['Cluster ID'] == clusters[i]] 
    a = dates_df.merge(a, how = 'left', on = 'Date')
    a = a.fillna(0)
    X[:,i,0:4] = a.iloc[:,2:6].values
    Y[:,i,0] = a.iloc[:,2].values
    
    a = OB_SH_F[OB_SH_F['Cluster ID'] == clusters[i]]
    a = dates_df.merge(a, how = 'left', on = 'Date')
    a = a.fillna(0)
    X[:,i,4:8] = a.iloc[:,2:6].values
    
    a = OB_LH_NF[OB_LH_NF['Cluster ID'] == clusters[i]]
    a = dates_df.merge(a, how = 'left', on = 'Date')
    a = a.fillna(0)
    X[:,i,8:12] = a.iloc[:,2:6].values
    Y[:,i,1] = a.iloc[:,2].values
    
    a = OB_LH_F[OB_LH_F['Cluster ID'] == clusters[i]]
    a = dates_df.merge(a, how = 'left', on = 'Date')
    a = a.fillna(0)
    X[:,i,12:16] = a.iloc[:,2:6].values
    
    a = IB_SH_NF[IB_SH_NF['Cluster ID'] == clusters[i]]
    a = dates_df.merge(a, how = 'left', on = 'Date')
    a = a.fillna(0)
    X[:,i,16:20] = a.iloc[:,2:6].values
    Y[:,i,2] = a.iloc[:,2].values
    
    a = IB_SH_F[IB_SH_F['Cluster ID'] == clusters[i]]
    a = dates_df.merge(a, how = 'left', on = 'Date')
    a = a.fillna(0)
    X[:,i,20:24] = a.iloc[:,2:6].values
    
    a = IB_LH_NF[IB_LH_NF['Cluster ID'] == clusters[i]]
    a = dates_df.merge(a, how = 'left', on = 'Date')
    a = a.fillna(0)
    X[:,i,24:28] = a.iloc[:,2:6].values
    Y[:,i,3] = a.iloc[:,2].values
    
    a = IB_LH_F[IB_LH_F['Cluster ID'] == clusters[i]]
    a = dates_df.merge(a, how = 'left', on = 'Date')
    a = a.fillna(0)
    X[:,i,28:32] = a.iloc[:,2:6].values
    
    X[:,i,32] = lat_long[lat_long['Cluster ID'] == clusters[i]].Latitude.values
    X[:,i,33] = lat_long[lat_long['Cluster ID'] == clusters[i]].Longitude.values

for i in range(T):
    a = FEMA[FEMA['Date'] == dates[i]]
    st = a['State'].unique()
    cl1 = cluster_state[cluster_state['State'].isin(st)]
    n_pos = cl1['Node'].unique()
    X[i,n_pos,34] = 1

for i in range(m):
    a = X[:,:,i]
    maxa = a.max()
    mina = a.min()
    X[:,:,i] = (a-mina)/(maxa-mina)
    
for i in range(l):
    a = Y[:,:,i]
    maxa = a.max()
    mina = a.min()
    Y[:,:,i] = (a-mina)/(maxa-mina)

X = X[0:T-1,:,:]
Y = Y[1:T,:,:]

np.save('X_Input_Daily', X)
np.save('Y_Output_Daily', Y)

