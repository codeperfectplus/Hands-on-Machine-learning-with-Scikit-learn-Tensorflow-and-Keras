import numpy as np
import pandas as pd
%matplotlib auto
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.linear_model import LinearRegression
plt.style.use("fivethirtyeight")


def getting_proper_cluster(x):
    pca=PCA(n_components=2)
    x=pca.fit_transform(x)
    C=[KMeans(n_clusters=i).fit(x) for i in range(1,10)]
    S=[C[i].score(x) for i in range(len(C))]
    plt.plot(S)
    return x


data1=pd.read_csv("realTraffic/realTraffic/speed_6005.csv")
data2=pd.read_csv("realTraffic/realTraffic/speed_7578.csv")
data3=pd.read_csv("realTraffic/realTraffic/speed_t4013.csv")
Data=pd.concat([data1,data2,data3],axis=0)

Data.plot(x="timestamp",y="value")
Data["timestamp"]=pd.to_datetime(Data["timestamp"])
Data["hour"]=Data["timestamp"].dt.hour
Data["day"]=Data['timestamp'].dt.dayofweek
Data["year"]=Data["timestamp"].dt.year
Data["month"]=Data["timestamp"].dt.month
Data["daylight"]=((Data["hour"]>=7) & (Data["hour"]>=22)).astype(int)
Data=Data.drop(["weekday"],axis=1)
Data["weekday"]=((Data["day"]<5)).astype(int)
Data=Data.drop(["timestamp"],axis=1)
V={}
for i in Data.columns:
    V[i]=Data[i].var()

Data=Data.drop(["year"],axis=1)
sns.heatmap(Data.corr(),annot=True,fmt=".2f")

G=Data.groupby(["value"]).agg([np.mean,np.var])

X=Data.values
X=getting_proper_cluster(X)

cluster=KMeans(n_clusters=3)
cluster.fit(X)
Data["PC1"]=X[:,0]
Data["PC2"]=X[:,1]

Data["category"]=cluster.predict(X)

number_of_categories=Data["category"].unique()
sns.boxplot(Data["category"],Data["value"])

for i in range(2):
    IF=IsolationForest(n_estimators=200,contamination=0.01)
    IF.fit(X[:,i].reshape(-1,1))
    Data[f"PC{i+1}_decision"]=IF.decision_function(X[:,i].reshape(-1,1))
    Data[f"PC{i+1}_anomaly"]=IF.predict(X[:,i].reshape(-1,1))
    


def scatter_plot(D):
    fig,ax=plt.subplots(2,1,figsize=(15,15))
    j=0
    for i in range(2):
        B=D.loc[D[f"PC{i+1}_anomaly"]==1,"value"]
        A=D.loc[D[f"PC{i+1}_anomaly"]==-1,"value"]
        ax[j].scatter(range(len(A)),A,c="r",label="anomaly")
        ax[j].scatter(range(len(B)),B,c="green",label="ok")
        if(j==0):
            ax[j].set_title("Principal_component 1",size=22)
        else:
            ax[j].set_title("Principal_component 2",size=22)
        j+=1
        plt.legend()
        plt.savefig("Anomaly_scatter.png")
        
        
def histo_plot(D):
    fig,ax=plt.subplots(2,1,figsize=(15,15))
    j=0
    for i in range(2):
        B=D.loc[D[f"PC{i+1}_anomaly"]==1,"value"]
        A=D.loc[D[f"PC{i+1}_anomaly"]==-1,"value"]
        ax[j].hist([A,B],color=["red","green"],label=["anomaly","normal"])
        
        if(j==0):
            ax[j].set_title("Principal_component 1",size=22)
        else:
            ax[j].set_title("Principal_component 2",size=22)
        j+=1
        plt.legend()
        plt.savefig("Anomaly_histogram.png")
        

histo_plot(Data)

def removing_outliers(D):
    data=D[((D["PC1_anomaly"]==1) & (D["PC2_anomaly"]==1))]
    return data


D=removing_outliers(Data)
X1=D[["hour","day","month","daylight","weekday","category"]].values
y=D["value"].values

# GBDT does not need feature scaling :)"
from sklearn.model_selection import cross_val_predict,train_test_split
classes=[DecisionTreeRegressor(max_depth=10),
GradientBoostingRegressor(max_depth=100,n_estimators=500,learning_rate=0.01),
LinearRegression(n_jobs=-1)]
Scores=[]
for i in classes:
    i.fit(X1,y)
    sc=cross_val_predict(i,X1,y,cv=5)
    sc=i.score(X1,sc)
    
    Scores.append(sc)


X_train,x_test,Y_train,y_test=train_test_split(X1,y,test_size=0.2,random_state=0)

R=GradientBoostingRegressor(max_depth=500,n_estimators=400,learning_rate=0.04)
R.fit(X_train,Y_train)

pred=R.predict(x_test)
from sklearn.metrics import mean_absolute_error
error=mean_absolute_error(y_test,pred)

# mae around 3












