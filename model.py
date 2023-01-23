import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
import pickle

# Load data
df = pd.read_csv("iris.csv")

# Select independent and dependent variables
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
y = df["Class"]

# Split the data set into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

# Instantiate the model
classifier = KNeighborsClassifier(leaf_size=1, p=2, n_neighbors=13)

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of the model
pickle.dump(classifier, open("model.pkl", "wb"))


''' The section of code below was used for hyperparameter tuning.
'''
# List of hyperparameters that we want to tune.
# leaf_size = list(range(1,50))
# n_neighbors = list(range(1,30))
# p=[1,2]

# Convert to dictionary
# hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

# Create new KNN object
# knn_2 = KNeighborsClassifier()

# Use GridSearch
# clf = GridSearchCV(knn_2, hyperparameters, cv=10)

# Fit the model
# best_model = clf.fit(X,y)

# Print The value of best hyperparameters
# print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
# print('Best p:', best_model.best_estimator_.get_params()['p'])
# print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])