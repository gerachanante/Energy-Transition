########## 0.IMPORTING & PREPARING DATA ##########
# 1.Importing the libraries
import pandas as pd
import numpy as np
import pydot as py

# 2.Importing the data
data = pd.read_excel('d.ETCleanDummyLM.xlsx')


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
        'LCOE Wind Onshore ($/MWh)','LCOE PV non-tracking ($/MWh)']
yhead = ['RE in TFEC (%)', 'CO2 per Capita (tCO2)', 'Energy Intensity (MJ/$2011 PPP GDP)',
         'RE IC per Capita (W)', 'Access to electricity (%)']

# 2.Adding random values column to compare with other variables
data['##random##'] = np.random.random(size=len(data))
  
# 3.Defining the values of X and Y
X = data[Xhead]
y = data[yhead]
  

########## 2.SPLITTING THE DATA TO TRAIN MACHINE LEARNING MODELS ##########
# 1.Splitting the data into the Training set and Test set
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 0)
      

########## 3.TRAINING THE MODELS ##########
# 1.Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 400, random_state = 0, 
                           max_depth=None, max_features = 'auto',
                           oob_score = True, bootstrap = True )
rf.fit(Xtrain, ytrain)
ypred = rf.predict(Xtest)
ypred_train = rf.predict(Xtrain)
print(rf)


########## 4.EVALUATING THE MODEL's PERFORMANCE ##########
# 1.Defining the performance metrics
from sklearn.metrics import mean_squared_error, r2_score
# evaluation for the training set
def evaluate_train(rf, Xtrain, ytrain):
    mse = mean_squared_error(ytrain, ypred_train)
    rmse = np.sqrt(mse)
    print("Model Performance on Training")
    print("%0.1f = Mean Squared Error"%(mse))
    print("%0.1f = RMSE"%(rmse))
# evaluation for the test set
def evaluate_test(rf, Xtest, ytest):
    mse = mean_squared_error(ytest, ypred)
    rmse = np.sqrt(mse)
    r2 = r2_score(ytest, ypred, multioutput='raw_values')
    print("Model Performance on Test")
    print("%0.1f = Mean Squared Error"%(mse))
    print("%0.1f = RMSE"%(rmse))
    print("Multi-output R2")
    print(pd.DataFrame(yhead,r2))
    
# 2.Calling the functions to evaluate the model's performance
evaluate_train(rf, Xtrain, ytrain)
#R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - ypred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum().
print("%0.3f = OOB R2 Score"%(rf.oob_score_))
evaluate_test(rf, Xtest, ytest)
print("%0.3f = Test Multi-output variance wt. R2"%(rf.score(Xtest, ytest)))


########## 5.FEATURE IMPORTANCES ##########
"""Feature importances are highly debateable. This script runs three methods
for calculating the importances of each X. 

1. Standard RF importances inside SkLearn (Gini Importance). These are 
calculated by averaging the drop in entropy for each X when it appears 
in decision tree nodes. The drop is from the parent branch to the child branch.
For regressions, this is the averaging of the drop in MSE for each X when it
appears in decision tree nodes

2. Permutated importances are calculated by comparing the RF baseline accuracy
metric (OOB Score = R2) vs each Xs drop in accuracy due to random values.
The RF is trained with all Xs, then replacing each X in the test set with 
random numbers and recalculating the RF's accuracy metric

3. Column drop importances are calculated by training the RF with all Xs and
storing its accuracy (OOB Score = R2) as a baseline score. Then, each column (X)
is dropped from the data and the model is retrained and rescored. Each score
is compared with the baseline and a substraction is stored. This is the true
importance of the variable since it completely takes it out of the analysis
each time.
"""
# 1.Extracting the SkLearn Gini Reduction Importance value from the features
impgini = rf.feature_importances_

# 2.Creating a dataframe with gini importances and features
impgini = pd.DataFrame(sorted(zip(Xhead, impgini), reverse=True),
                  columns=['Feature','Variance Importance'])

# 3.Extracting the permutation importances 
from rfpimp import importances
impperm = importances(rf, Xtest, ytest)

# 4.Developing the drop column importances
from sklearn.base import clone
# Defining the importances function
def dropcol_importances(rf, Xtest, ytest):
    rf_ = clone(rf)
    # Defining a random state so the "cloning" works
    rf_.random_state = 999
    # Fitting the cloned RF to the training set
    rf_.fit(Xtrain, ytrain)
    # Defininig baseline as the standard R2 score of the model
    baseline = rf_.oob_score_
    imp = []
    for col in Xtrain.columns:
        X = Xtrain.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.random_state = 999
        rf_.fit(X, ytrain)
        # Defining the o as the R2 score for each dropped column model
        o = rf_.oob_score_
        # Importance as the standard score - each model score
        imp.append(baseline - o)
    imp = np.array(imp)*100
    I = pd.DataFrame(
            data={'Feature':Xtrain.columns,
                  'Drop Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Drop Importance', ascending=False)
    return I
# Calling the function for drop importance
"""impdrop = dropcol_importances (rf, Xtest, ytest)
"""
# 5.Combining the calculated importances as a single dataframe
# Merging the dataframes, first Gini and Permutation importances
imp = pd.merge(impgini, impperm, on='Feature', how='outer')
# Then adding the drop importance - not to be used for a faster model
"""imp = pd.merge(imp, impdrop, on='Feature', how='outer')
"""
# renaming columns
imp.columns = ['Influencer','Variance Importance','Permutation Importance']
"""imp.columns = ['Influencer','Variance Importance','Permutation Importance','Drop Column Importance']
"""
# sorting ascending on Variance Importances
imp = imp.sort_values(by=['Variance Importance'],ascending = True)


########## 6.EXPORTING THE IMPORTANCES ##########
imp.to_excel(r'D:\OneDrive - International Renewable Energy Agency - IRENA\05.Master Thesis\05.Analysis\02.RF Feature Impotances\d.ET.RF ResultsLower Middle Income.xlsx', 
               sheet_name='Sheet1', index=False)


########## 7. VISUALIZING ONE OF THE TREES OF THE FOREST ##########
# 1.Importing tools needed for visualization
from sklearn.tree import export_graphviz
# 2.Pulling out one tree from the forest
tree = rf.estimators_[5]
# 3.Exporting the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = Xhead, 
                rounded = True, precision = 1, max_depth = 2)
# 4.Using dot file to create a graph
(graph, ) = py.graph_from_dot_file('tree.dot')
# 5.Saving graph as a png file
graph.write_png('tree.png')
# 6.Saving graph as a pdf file
graph.write_pdf('tree.pdf')

