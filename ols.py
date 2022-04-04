'''
This Python script computes models for property values based on a dataset specified
in sales.csv in the working directory. It then outputs the model type, variables, and
associated MSE into Results.csv, as well as in console output.

Created by Cody Olivotto and Adeeb Salim for ECO4444. 
'''

# import necessary libraries 
import numpy as np
from itertools import combinations
from pandas import read_csv
from pandas import DataFrame
from numpy.random import seed
from sklearn.model_selection import cross_validate
import statsmodels.formula.api
from sklearn import preprocessing

#read in data from sales.csv
data = read_csv('./sales.csv', delimiter=',')

#set seed for random to 1234 for equal comparison reasons
seed(1234)
#randomize data order from seed
data = data.sample(len(data))

# set dependent var as price, and assign input vars
y = data['Price']
x = data[['Year', 'Home_size', 'Parcel_size', 'Beds', 'Age', 'Pool', 'X_coord', 'Y_coord']]
'''
---------- Determine all combinations of (varCount - 1) variables -------------
'''
baseVarList = ['Year', 'Home_size', 'Parcel_size', 'Beds', 'Age',\
                                           'Pool', 'X_coord', 'Y_coord']
x_combos = [] # create blank combo list

#create list of tuple pairs of variables for polynomial features
polypairs = list(combinations(baseVarList, 2))
polyVars = []
for i in polypairs:
    newmult = i[0] + '*' + i[1]
    x[newmult] = (x[i[0]] * x[i[1]])
    polyVars.append(newmult)
#get squares of base vars
for i in baseVarList:
    newSquare = i+'**2'
    x[newSquare] = x[i]**2
    polyVars.append(newSquare)
#get a total variable list
totalVars = []
totalVars.extend(baseVarList)
totalVars.extend(polyVars)

#get combos of every variable and their polys up to (varCount - 1) variables per model
varCount = 3
for n in range(1, varCount):
    combos = combinations(totalVars, n)
    x_combos.extend(combos)

#always include full set of base var combos regardless of var limit
for n in range(varCount, 9):
    combos = combinations(baseVarList, n)
    x_combos.extend(combos)

'''
--------------------------------- Scale Dataset -------------------------------
'''
x_scaled = preprocessing.scale(x)
x = DataFrame(x_scaled, columns = totalVars)
'''
--------------------------------- Model Calculations --------------------------
'''
# store models
ols_models = {}

# OLS Method w/ kfold
print("Now running OLS modeling iterations...")
for n in range(0, len(x_combos)):
    combo_list = list(x_combos[n])
    xinput = x[combo_list]
    model = statsmodels.api.OLS(y, xinput)
    cv_scores = cross_validate(model, xinput, y, cv=10, scoring=('neg_mean_squared_error'),
        return_estimator=True, n_jobs=2)
    #print(np.mean(cv_scores['estimator']))
    ols_models[n] = np.mean(cv_scores['test_score'])

# print out the lowest Test MSE for OLS
print("Outcomes from the best OLS regression Model:")
min_mse = abs(max(ols_models.values()))
print("Minimum Avg Test MSE:", min_mse.round(2))
for possibles, i in ols_models.items():
    if i == -min_mse:
        print("The Combination of Variables:", str(x_combos[possibles]))
        bestVarList = list(x_combos[possibles])
#save best OLS model to array for ease of access
best_ols_model = np.array(['Linear', str(bestVarList), min_mse])
