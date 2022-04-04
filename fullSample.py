'''
By Cody Olivotto, and Adeeb Salim
'''
################ Library ######################################################
import statsmodels.formula.api as smf
import numpy as np
from pandas import read_csv
from numpy.random import seed
from pandas import DataFrame


################ Data Import and radnomization ###############################
data = read_csv('./sales.csv', delimiter=',')

seed(1234)
data = data.sample(len(data))

  
################ Hardcoded Variables for the Model ###########################
var0 = data['Price'] # Y-Variable
var1 = data['Beds']
var2 = data['Beds']*data['Year']
var3 = data['Parcel_size']*data['Pool']
var4 = pow(data['Home_size'],2)


################ Preparing the data ##########################################
x = np.array([var0, var1, var2, var3, var4])
x = np.transpose(x)
z = DataFrame(x, columns=('Price', 'Beds', 'BedxYear', 'ParcelxPool', 'Home_sizeSQR'))

################ Running the Model (With 'R' style code ######################
model = smf.ols('Price ~ Beds + BedxYear + ParcelxPool + Home_sizeSQR', data = z)
results = model.fit()


################ Save and print Model information ###########################
full_mse = np.array(round(results.mse_resid,4))
full_r2 = np.array(round(results.rsquared,4))
Exog = np.array('Beds,  BedxYear , ParcelxPool , Home_sizeSqr')
print(results.summary())
print ("MSE Full Sample: ", full_mse)
print("R-Squared Full Sample", full_r2)

################ Exporting the Full Sample Results to CSV ####################
full_results = np.array([Exog, full_mse, full_r2])
full_results = DataFrame(full_results)
full_results.to_csv('./ResultsTable2.csv')




















































