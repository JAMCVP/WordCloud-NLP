import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


#IMPORTING DATA
data = load_breast_cancer()
#print(data)
data_cancer = pd.DataFrame(np.c_[data['data'],data['target']], columns= np.append(data['feature_names'],['target']))
#print(data_cancer.info())
#print(data_cancer.head())

#DATA VISUALIZATION
#sns.pairplot(data_cancer, vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness'])
sns.pairplot(data_cancer,hue= 'target', vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area','mean smoothness'])
plt.show()
sns.countplot(data_cancer['target'])
plt.show()
sns.scatterplot(x='mean area', y= 'mean smoothness', hue= 'target', data = data_cancer)
plt.show()
plt.figure(figsize=(25,25))
sns.heatmap(data_cancer.corr(), annot= True)
plt.show()


#MODELING

X = data_cancer.drop(['target'], axis=1)
y= data_cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5)
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)
svc_model = SVC()
svc_model.fit(X_train, y_train)
predictions= svc_model.predict(X_test)
cm= confusion_matrix(y_test,predictions)
sns.heatmap(cm, annot=True)
plt.show()
print(classification_report(y_test,predictions))


