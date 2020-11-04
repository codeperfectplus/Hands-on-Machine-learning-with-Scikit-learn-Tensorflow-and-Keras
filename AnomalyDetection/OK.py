import pandas as pd
import numpy as np
%matplotlib auto
import matplotlib
import seaborn as sns
import matplotlib.dates as md
from matplotlib import pyplot as plt

from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

data=pd.read_csv("artificialWithAnomaly/artificialWithAnomaly/art_daily_flatmiddle.csv")
data["timestamp"]=pd.to_datetime(data["timestamp"])
sns.distplot(data["value"])
data["HR"]=data["timestamp"].dt.hour
data.plot(x="timestamp",y="value")
data["day"]=data["timestamp"].dt.day
An=data[((data["day"]>=11 ) & (data["day"]<=12))]
An.plot(x="timestamp",y="value")

data["dayOfWeek"]=data["timestamp"].dt.dayofweek
data["weekday"]=(data["dayOfWeek"]<5).astype(int)
data["year"]=data["timestamp"].dt.year

data["day/night"]=((data["HR"]>=7) & (data["HR"]<=22)).astype(int)
VAR={}
for i in data.columns[1:]:
    VAR[i]=data[i].var()
    
for j in VAR:
    if VAR[j]==0.0:
        drop_c=j

data=data.drop([drop_c],axis=1)
data=data.drop(["timestamp"],axis=1)

X=StandardScaler().fit_transform(data.values)
pca=PCA(n_components=3)
X=pca.fit_transform(X)

A=range(1,20)


print(__doc__)
Kmean=[KMeans(n_clusters=i).fit(X) for i in A]
elbow_score=[Kmean[i].score(X) for i in range(len(Kmean))]
plt.plot(A,elbow_score)


clus=KMeans(n_clusters=4).fit(X)
clus.cluster_centers_
data["classes"]=clus.predict(X)
data["P1"]=X[:,0]
data["P2"]=X[:,1]
data["classes"].plot.hist()
data["classes"].nunique()
Color={0:"red",1:"blue",2:"green",3:"pink"}
fig=plt.figure(figsize=(15,12))
plt.scatter(data["P1"],data["P2"],c=data["classes"].apply(lambda x:Color[x]))
Dis=pd.Series()
for i in range(len(data)):
    a=X[i,:]
    b=clus.cluster_centers_[clus.labels_[i]-1]
    Dis.set_value(i,np.linalg.norm(a-b))

Ds=int(0.01*len(Dis))
D=Dis.nlargest(Ds).min()
data["anomaly"]=(Dis>=D).astype(int)
Ac={1:"red",0:"green"}
plt.scatter(data["P1"],data["P2"],c=data["anomaly"].apply(lambda x:Ac[x]))


A=data.loc[data["anomaly"]==0,"value"]
B=data.loc[data["anomaly"]==1,"value"]
plt.hist([A,B],bins=10,stacked=True,color=["blue","red"],label=["normal","anomaly"])
plt.legend()










    
    
    





















