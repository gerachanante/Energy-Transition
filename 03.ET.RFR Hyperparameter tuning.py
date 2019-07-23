########## 0.IMPORTING & PREPARING DATA ##########
# 1.Importing the libraries
import pandas as pd
import numpy as np

# 2.Importing the data
data = pd.read_excel('d.ETCleanDummy.xlsx')


########## 1.DEFINING X & Y ##########
# 1.Defining the titles of X and of Y
Xhead = ['HDI','Government Effectiveness (+-2.5)','Regulatory Quality (+-2.5)',
        'Time to electricity (days)',
        'GDP per capita, PPP ($current)','Renewable freshwater (bm3)',
        'Non-RE Net Decom. 2000 (MW)',
        'Median Time Commitment-Commissioning (days)','Access to clean cooking fuel & tech (%)',
        'Strategic planning','Economic Instruments',
        'Policy Support','Technology deployment and diffusion','Regulatory Instrument',
        'Loans','Information provision','Information and Education','Voluntary Approaches',
        'Grants and subsidies','GHG emissions trading','Comparison label','Obligation schemes',
        'Building codes and standards','Energy imports, net (% of energy use)',
        'CO2 intensity (kg per kgoe energy use)','Policies','FIT/Premiums',
        'Performance label','Technology deployment','RD&D',
        'PM2.5 Avg Concentration (ug/m3)','Gasoline Pump Price ($/L)',
        'Start-up Procedures Cost (% of GNI per capita)','Time to start-up (days)',
        'GNI per capita, PPP ($current)','R&D expenditure (% of GDP)',
        'Committed per project ($2016M)','Year','RE per Auction (MW)',
        'CapEx per IC ($/W) - Hydropower','CapEx per IC ($/W) - Solid biofuels and renewable waste',
        'CapEx per IC ($/W) - Biogas','CapEx per IC ($/W) - Onshore wind energy',
        'CapEx per IC ($/W) - Solar photovoltaic','Crude oil price ($2017/barrel)',
        'Avg Natural Gas price ($/MBTU)','Avg Coal Price ($/tonne)','Avg NPP (gC/m2/y)',
        'Avg PV Out (kWh/kWp/y)','Avg Wind Power Density (W/m2)','Median Power Price ($/MWh)',
        'LCOE Wind Onshore ($/MWh)','LCOE PV non-tracking ($/MWh)','WBG Income Group_High income',
        'WBG Income Group_Low income','WBG Income Group_Lower middle income',
        'WBG Income Group_Upper middle income','##random##']
yhead = ['RE in TFEC (%)', 'GHG per capita (tCO2)', 'Energy Intensity (MJ/$2011 PPP GDP)',
         'RE IC per Capita (W/person)', 'Access to electricity (%)']

# 2.Adding random values column to compare with other variables
data['##random##'] = np.random.random(size=len(data))
  
# 3.Defining the values of X and Y
X = data[Xhead]
y = data[yhead]


########## 2.SPLITTING THE DATA TO TRAIN MACHINE LEARNING MODELS ##########
# Splitting the data into the Training set and Test set
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 0)


########## 3.TRAINING THE MODELS ##########
# 1.Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 0, oob_score = True, bootstrap = True )


########## 4.TUNING THE HYPER-PARAMETERS WITH RANDOM GRID CV ##########
# 1.Importing libraries for a random grid of parameters to cross-validate
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

# 2.Creating the random grid
random_grid = {'max_features': ['auto', 'sqrt', 'log2', None],
               'max_depth': [int(x) for x in np.linspace(start = 5, stop = 300, num = 20)],
               'min_samples_split': [int(x) for x in np.linspace(start = 2, stop = 100, num = 20)],
               'min_samples_leaf': [int(x) for x in np.linspace(start = 1, stop = 100, num = 20)]
               }
pprint(random_grid)
"""'n_estimators': [int(x) for x in np.linspace(start = 400, stop = 400, num = 1)],
               """
# 3.Searching for the best hyper-parameters with cross-validation
# n_iter - how many combinations
# cv - how many cross-validations for each combination
# scoring - how to determine the score for each job (MSE used)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               n_iter = 400, cv = 3, verbose=2, random_state=0, 
                               n_jobs = -1, scoring='neg_mean_squared_error')

# 4.Fitting the random search model to the training data
rf_random.fit(Xtrain, ytrain)

# 5.Defining RMSE for each parameter configuration
for mean_score, params in zip(rf_random.cv_results_["mean_test_score"],
                              rf_random.cv_results_["params"]):
    print(np.sqrt(-mean_score), params)
    
# 6.Showing the model with the least RMSE
# best model
rf_random.best_estimator_
# best combo of parameters of random search
print('use these parameters for the model', rf_random.best_params_)


########## 5. TESTING THE BEST PARAMETERS ON TRAINING SET ##########
# 1.Defining the performance metrics (MAPE)
# predicted y values from the best paremeters
ytrain_random = rf_random.best_estimator_.predict(Xtrain)
# absolute error between predicted y and real y
ertrain = abs(ytrain_random - ytrain)
# mean absolute percentage error (MAPE) = average ( absolute error / real y)
mape = np.mean(100 * (ertrain / ytrain))

# 2.Calculating and showing model accuracy and RMSE
# Accuracy is 100 minus the MAPE
accuracy = 100 - mape    
print('The best model from the randomized search predicts on the training set with an accuracy of', 
      round(accuracy, 1),'%')
# RMSE is the squared root of the MSE between the predicted y and the real y
from sklearn.metrics import mean_squared_error
msetrain = mean_squared_error(ytrain, ytrain_random)
rmsetrain = np.sqrt(msetrain)
print('The best model from the randomized search predicts on training set with a Root Mean Squared Error (RMSE) of', 
      round(rmsetrain, 2))


########## 6. TESTING THE BEST PARAMETERS ON TEST SET ##########
# 1.Defining the performance metrics (MAPE)
# predicted y values from the best paremeters
ypred_random = rf_random.best_estimator_.predict(Xtest)
# absolute error between predicted y and real y
ertest = abs(ypred_random - ytest)
# mean absolute percentage error (MAPE) = average ( absolute error / real y)
mape = np.mean(100 * (ertest / ytest))

# 2.Calculating and showing model accuracy and RMSE
# Accuracy is 100 minus the MAP
accuracy = 100 - mape   
print('The best model from the randomized search predicts on the test set an accuracy of', 
      round(accuracy, 1),'%')
# RMSE is the squared root of the MSE between the predicted y and the real y
msetest = mean_squared_error(ytest, ypred_random)
rmsetest = np.sqrt(msetest)
print('The best model from the randomized search predicts on test set with a Root Mean Squared Error (RMSE) of',
      round(rmsetest,1))


 

