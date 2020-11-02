import numpy as np
import pandas as pd
%matplotlib auto
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans
from sklearn.
plt.style.use("fivethirtyeight")
fig=plt.figure(figsize=(12,15))
data=pd.read_csv("Social_Network_Ads.csv")
data=data.iloc[:,2:]

treeclass=RandomForestClassifier(n_estimators=100,max_depth=10)
X,y=data.iloc[:,:-1].values,data.iloc[:,-1].values
def plotting_decision_(X,Y,CL):
    X=StandardScaler().fit_transform(X)
    X_train,x_test,Y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    xx_min,xx_max=X[:,0].min()-0.5,X[:,0].max()+0.6
    xx,yy=np.meshgrid(np.arange(xx_min,xx_max,0.2),np.arange(xx_min,xx_max,0.2))
    cmap_bright=ListedColormap(["red","azure"])
    cl=CL()
    cl.fit(X_train,Y_train)
    score=cl.predict(x_test)
    Z=cl.decision_function(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)
    plt.contour(xx,yy,Z,cmap=plt.cm.jet)
    plt.scatter(X_train[:,0],X_train[:,1],c=Y_train,cmap=cmap_bright)
    plt.text(xx.max()-.3,xx.min()+.3,(np.mean(score)),size=15,horizontalalignment="right")
    
#sns.relplot(x="Age",y="EstimatedSalary",data=data,hue="Purchased")
#sns.boxplot(x=data["Purchased"],y=data["EstimatedSalary"],whis=2,saturation=0.6)
#from sklearn.ensemble import IsolationForest
#IF=IsolationForest(n_estimators=100,bootstrap=False)
#IF.fit(X[:,0].reshape(-1,1))
#xx=np.linspace(X[:,0].min()-5,X[:,0].max()+5,len(data)).reshape(-1,1)
#outlier=IF.predict(xx)
#anomaly_score=IF.decision_function(xx)
#plt.plot(xx,anomaly_score,label="automated")
    
    






            









