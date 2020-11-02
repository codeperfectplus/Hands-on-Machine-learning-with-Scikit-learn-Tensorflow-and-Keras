from  __future__ import print_function
import numpy as np
import pandas as pd
%matplotlib auto
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import *
from sklearn.ensemble import VotingClassifier,RandomForestClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from xgboost import XGBoost
data=pd.read_csv("Social_Network_Ads.csv")
data=data.iloc[:,2:]
from sklearn.utils import shuffle
shuffle(data)
cl=VotingClassifier(estimators=[("lr",LogisticRegression()),("ran",RandomForestClassifier(n_estimators=300,max_depth=20)),("svm",SVC(probability=True))],voting="soft")
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
X=StandardScaler().fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

cl.fit(X_train,Y_train)
score=cl.score(x_test,y_test)
L=SVC(kernel="rbf",probability=True)
PAS=BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,max_samples=100,bootstrap=False,
                     n_jobs=-1)
BG=BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,max_samples=100,bootstrap=True,
                     n_jobs=-1)

def plot_dec(X,y,CL):
    fig=plt.figure(figsize=(15,20))
    X_train,x_test,Y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
    xx_min,xx_max=X[:,0].min()-0.5,X[:,0].max()+0.5
    yy_min,yy_max=X[:,1].min()-0.5,X[:,1].max()+0.5
    xx,yy=np.meshgrid(np.arange(xx_min,xx_max,0.2),np.arange(yy_min,yy_max,0.2))
    cm_bright=ListedColormap(["red","azure"])
    CL.fit(X_train,Y_train)
    score=CL.score(x_test,y_test)
#    OOBS=CL.oob_score_
#    print(OOBS)
    if hasattr(CL,"decision_function"):
        Z=CL.decision_function(np.c_[xx.ravel(),yy.ravel()])
        print(Z)
    else:
        Z=CL.predict_proba(np.c_[xx.ravel(),yy.ravel()])
        print(Z)
        Z=Z[:,1]
    
    Z=Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z)
    plt.scatter(X_train[:,0],X_train[:,1],c=Y_train,cmap=cm_bright)
    plt.text(xx.max()-0.3,yy.min()+0.3,(score),horizontalalignment="right",size=25)
    #plt.text(xx.max()-0.5,yy.min()+0.9,(OOBS),horizontalalignment="left",size=25)

    return Z
    
Z=plot_dec(X,y,BG)
Z=plot_dec(X,y,PAS)
Z=plot_dec(X,y,DecisionTreeClassifier())

X=np.array([12,34,56])
np.arange(X.min(),X.max()+1,0.3)

def random_sampling_data(data):
    test_set_length=int(len(data)//5)
    train_set_length=int(len(data)//(5/4))
    test_idx=np.random.randint(0,len(data)-1,size=(test_set_length,),dtype=np.int32)
    train_idx=np.random.randint(0,len(data)-1,size=(train_set_length,),dtype=np.int32)
    X_train=[]
    x_test=[]
    for i in test_idx:
        x_test.append(data.iloc[i,:-1].values)
    for j in train_idx:
        X_train.append(data.iloc[j,:-1].values)
        
    return np.asarray(X_train),np.asarray(x_test)

classes=[DecisionTreeClassifier(),RandomForestClassifier(max_depth=100,n_estimators=300)
            ,SVC(kernel="linear"),SVC(kernel="rbf",gamma=0.3,C=4.0),
            BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,max_samples=100,bootstrap=False,
                     n_jobs=-1),
               BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,max_samples=100,bootstrap=True,
                     n_jobs=-1),
               AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=500,algorithm="SAMME.R",learning_rate=0.05)                  
                       ]


from sklearn.model_selection import cross_val_predict
A=[]
for i in classes:
    i.fit(X,y)
    scores=cross_val_predict(i,X,y,cv=5)
    score=i.score(X,scores)
    A.append(score)
    




