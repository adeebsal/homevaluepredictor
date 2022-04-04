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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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
    
baseCombos = []

for n in range(1, 9):
    combos = combinations(baseVarList, n)
    baseCombos.extend(combos)


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
ridge_models = {}
lasso_models = {}

# OLS Method w/ kfold
print("Now running OLS modeling iterations...")
model = LinearRegression()
xinput = x[['Beds', 'Year*Beds', 'Parcel_size*Beds', 'Home_size**2']]
cv_scores = cross_validate(model, xinput, y, cv=10, scoring=('neg_mean_squared_error', 'r2'),
    return_estimator=True, return_train_score=(True), n_jobs=2)
ols_models[0] = np.mean(cv_scores['test_neg_mean_squared_error'])

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
ols_r_sqd_train = np.mean(cv_scores['train_r2']) #### I think these should be aboslute values, check after model finished running
ols_r_sqd_test = np.mean(cv_scores['test_r2'])
ols_mse_train = -np.mean(cv_scores['train_neg_mean_squared_error'])

#set up alpha range for ridge and lasso
alphas = range(1, 50)
alphadict = {}

#Run ridge regression using best OLS variables
print("Now running ridge regression over alpha with all base variable combos...")
model = Ridge()
xinput = x[['Year', 'Home_size', 'Parcel_size', 'Beds', 'Age', 'Y_coord']]
for a in alphas:
    model.set_params(alpha = a)
    ridge_cv_scores = cross_validate(model, xinput, y, cv=10, 
        scoring=('neg_mean_squared_error', 'r2'), return_train_score=True, 
            return_estimator=(True), n_jobs=2)
    alphadict[a] = np.mean(ridge_cv_scores['test_neg_mean_squared_error'])
    lowestalpha = min(alphadict, key=alphadict.get)
ridge_models[0] = alphadict[lowestalpha]
#Print lowest MSE for Ridge
ridge_r_sqd_train = np.mean(ridge_cv_scores['train_r2']) #### I think these should be aboslute values, check after model finished running
ridge_r_sqd_test = np.mean(ridge_cv_scores['test_r2'])
ridge_mse_train = -np.mean(ridge_cv_scores['train_neg_mean_squared_error'])

print("Outcomes from the best Ridge Regression model:")
min_mse = abs(max(ridge_models.values()))
print("Minimum Avg Test MSE:", min_mse.round(2))
for possibles, i in ridge_models.items():
    if i == -min_mse:
        print("Alpha with lowest MSE:", lowestalpha)
        poss = list(baseCombos[possibles])
#save best ridge to array for easy access
best_ridge_model = np.array(['Ridge', str(poss) , min_mse, lowestalpha])

#Run lasso using best OLS variables
print("Now running lasso over alpha with all base variable combos...")
model = Lasso()
xinput = x[['Year', 'Home_size', 'Parcel_size', 'Beds', 'Age', 'Pool', 'X_coord', 'Y_coord']]
for a in alphas:
    model.set_params(alpha = a)
    lasso_cv_scores = cross_validate(model, xinput, y, cv=10, scoring=('neg_mean_squared_error', 'r2'), 
        return_train_score=True, return_estimator=(True), n_jobs=2)
    alphadict[a] = np.mean(lasso_cv_scores['test_neg_mean_squared_error'])
    lowestalpha = min(alphadict, key=alphadict.get)
lasso_models[0] = alphadict[lowestalpha]

#Print lowest MSE for Lasso
lasso_r_sqd_train = np.mean(lasso_cv_scores['train_r2'])
lasso_r_sqd_test = np.mean(lasso_cv_scores['test_r2'])
lasso_mse_train = -np.mean(lasso_cv_scores['train_neg_mean_squared_error'])

print("outcomes from the Best Lasso Regression Model:")
min_mse = abs(max(lasso_models.values()))
print("minimum Avg Test MSE:", min_mse.round(2))
for possibles, i in lasso_models.items():
    if i == -min_mse:
        print("Alpha with lowest MSE:", lowestalpha)
        poss = list(baseCombos[possibles])
#save best lasso to array for easy access        
best_lasso_model = np.array(['Lasso', str(poss) , min_mse, lowestalpha])



'''
-------------------------- Final Results reporting ----------------------------
'''
#master_results = np.array([best_lasso_model, best_ols_model, best_ridge_model])
#master_results = DataFrame(master_results, columns=("Model", "Predictors", 'MSE', 'Alpha'))

#master_results.to_csv('./Results.csv')