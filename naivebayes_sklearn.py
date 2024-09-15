from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np



X=np.array(
    [
        [0,1.6],
        [1,2],
        [0,1.9],
        [0,1.88],
        [0,1.7],
        [1,1.85],
        [0,1.6],
        [1,1.7],
        [1,2.2],
        [1,2.1],
        [0,1.8],
        [1,1.95],
        [0,1.9],
        [0,1.8],
        [0,1.75]
        ]
    )
        
        
print("X ",X)
Y=np.array(
    [
        "S",
        "T",
        "M",
        "M",
        "S",
        "M",
        "S",
        "S",
        "T",
        "T",
        "M",
        "M",
        "M",
        "M",
        "M",
        ]
    )
               
print("Y ",Y)


X_train,X_test,Y_train ,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print(X_train)
clf=GaussianNB()
clf.fit(X_train,Y_train)

Y_pred=clf.predict(X_test)
my_pd=clf.predict(np.array([[1,1.95]]))
print("My prediction :",my_pd)
accuracy=accuracy_score(Y_test,Y_pred)
print("Accuracy:",accuracy)
