'''
This script calculates the final predictions and MSE using
the best model we found using modelfinder.py. Model is hardcoded
into this script.
'''
# import necessary libraries 
import statsmodels.formula.api as smf
import numpy as np
from pandas import read_csv
from numpy.random import seed
from pandas import DataFrame
from sklearn.metrics import mean_squared_error

#read in data from valset.csv
data = read_csv('./validation_set.csv', delimiter=',')

#set seed for random to 1234 for equal comparison reasons
seed(1234)
#randomize data order from seed
data = data.sample(len(data))

'''
--------------------------------- Data setup --------------------------
'''
  
############## Hardcoded Variables for the Model ###########################
var0 = data['price'] # Y-Variable
var1 = data['year']*(data["home_size"]**2)
var2 = (data['parcel_size']**2)*data['home_size']
var3 = data['parcel_size']*data['home_size']

################ Preparing the data ##########################################
x = np.array([var0, var1, var2, var3])
x = np.transpose(x)
z = DataFrame(x, columns=('Price', 'YearxHome_size2', 'Home_sizexParcel_size2', 'Home_sizexParcel_size'))
'''
--------------------------------- Model Calculations --------------------------
'''

from joblib import dump, load

model = load('./ols.model')

predictedPrices = model.predict(z)

print(predictedPrices)

MSE = mean_squared_error(z['Price'], predictedPrices)
print("MSE of the model on the validation set: ", MSE)