# before doing the hyperparameter tuning. we can use lazypredict's LazyClassifier to find the best classifier visit(https://lazypredict.readthedocs.io/en/latest/usage.html#classification)
# place the code in src/models/ and name it best_models.py
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
data = load_breast_cancer() # change the data with your data set
X = data.data
y= data.target
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
models # give list of best models from best to worst
models.iloc[1,:] # give details of the best model

# find out how to get the name of the best model from the models variable above

# get the best model and store it in x like 
x = 'RandomForestRegressor'
eval(x)() # it is same as RandomForestRegressor

# now we can use it to do the hyper parameter tuning in the scr/models/train_model.py