import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load data
data = pd.read_csv('Iris.csv')
# print(data.head())
X = data.iloc[:, 1:5].values
# print(X)
Y= data.iloc[:,-1].values
# print(Y)

# split data
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,random_state=0)

# build tree
dt_model = tree.DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
text_representation = tree.export_text(dt_model)
print(text_representation)
features = data.columns[1:5]
print(features)
class_name = data.Species.unique()
print(class_name)
fig=plt.figure(figsize=(10,8))
_ = tree.plot_tree(dt_model,feature_names=features,class_names=class_name,filled=True)
plt.show()

# predict
predictedvalues = dt_model.predict(x_test)

# evaluate model
Accuracy = accuracy_score(y_test,predictedvalues)
print("The Accuracy of the model = ",Accuracy)