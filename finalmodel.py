'''
This script generates the final fitted model based on the variable combinations
found to work best using consolidated.py. Saves model to ols.model to be imported
into other scripts.
'''
# import necessary libraries 
import statsmodels.formula.api as smf
import numpy as np
from pandas import read_csv
from numpy.random import seed
from pandas import DataFrame

#read in data from sales.csv
data = read_csv('./sales.csv', delimiter=',')

#set seed for random to 1234 for equal comparison reasons
seed(1234)
#randomize data order from seed
data = data.sample(len(data))

'''
--------------------------------- Model Calculations --------------------------
'''
  
################ Hardcoded Variables for the Model ###########################
var0 = data['Price'] # Y-Variable
var1 = data['Year']*(data["Home_size"]**2)
var2 = (data['Parcel_size']**2)*data['Home_size']
var3 = data['Parcel_size']*data['Home_size']

################ Preparing the data ##########################################
x = np.array([var0, var1, var2, var3])
x = np.transpose(x)
z = DataFrame(x, columns=('Price', 'YearxHome_size2', 'Home_sizexParcel_size2', 'Home_sizexParcel_size'))

################ Running the Model (With 'R' style code ######################
model = smf.ols('Price ~ YearxHome_size2 + Home_sizexParcel_size2 + Home_sizexParcel_size', data = z)
results = model.fit()

################ Save and print Model information ###########################
full_mse = np.array(round(results.mse_resid,4))
full_r2 = np.array(round(results.rsquared,4))
Exog = np.array('YearxHome_size2, Home_sizexParcel_size2, Home_sizexParcel_size')
print(results.summary())
print ("MSE Full Sample: ", full_mse)
print("R-Squared Full Sample", full_r2)

from joblib import dump
dump(results, './ols.model')