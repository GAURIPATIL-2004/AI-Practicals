import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier, plot_tree  
from sklearn import metrics  

# Load dataset and convert to DataFrame
iris = load_iris()  
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)  
data['Species'] = iris.target  
data['Species'] = data['Species'].replace(dict(zip(np.unique(iris.target), np.unique(iris.target_names))))  

# Split dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data.drop("Species", axis=1), data['Species'], test_size=0.3, random_state=93)

# Fit decision tree model
dtc = DecisionTreeClassifier(max_depth=3, random_state=93)  
dtc.fit(x_train, y_train)  

# Plot Decision Tree
plt.figure(figsize=(20, 7))  
plot_tree(dtc, feature_names=iris.feature_names, class_names=dtc.classes_, rounded=True, filled=True, fontsize=10)  
plt.show()  

# Predict and create confusion matrix
y_pred = dtc.predict(x_test)  
conf_matrix = metrics.confusion_matrix(y_test, y_pred)  

# Plot heatmap of confusion matrix
plt.figure(figsize=(10, 7))  
sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="magma", fmt="g", xticklabels=dtc.classes_, yticklabels=dtc.classes_)  
plt.title('Confusion Matrix')  
plt.xlabel("Predicted Values")  
plt.ylabel("True Labels")  
plt.show()