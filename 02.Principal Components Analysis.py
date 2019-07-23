########## 0.IMPORTING & PREPARING DATA ##########
# 1.Importing the libraries
import pandas as pd
import numpy as np

# 2.Importing the data
data = pd.read_excel('d.ETCleanScaledDummy.xlsx')


########## 1.DEFINING X & Y ##########
# 1.Defining the titles of X and of Y
Xhead = ['HDI','Government Effectiveness (+-2.5)','Regulatory Quality (+-2.5)',
        'Time to electricity (days)',
        'GDP per capita, PPP ($current)','Renewable freshwater (bm3)',
        'Non-RE Net Decom. 2000 (MW)',
        'Median Time Commitment-Commissioning (days)','Access to clean cooking fuel & tech (%)',
        'Pre-existing RE IC (MW)','Strategic planning','Economic Instruments',
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
        'WBG Income Group_Upper middle income']
yhead = ['RE in TFEC (%)', 'GHG per capita (tCO2)', 'Energy Intensity (MJ/$2011 PPP GDP)',
           'RE IC per Capita (W/person)', 'Access to electricity (%)']

# 2.Defining the values of X and Y
X = data[Xhead]
y = data[yhead]


########## 2.SPLITTING THE DATA TO TRAIN MACHINE LEARNING MODELS ##########
# Splitting the data into the Training set and Test set
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 0)


########## APPLYING PRINCIPAL COMPONENTS ANALYSIS ##########
from sklearn.decomposition import PCA
#None works for explained variance for all principle components
pca = PCA(n_components = None) 
Xtrain = pca.fit_transform(Xtrain)
Xtest = pca.transform(Xtest)
explained_variance = pca.explained_variance_ratio_ #lists all PCA with their variance. Then select the top 2 for replacing the Xs

# E. FIT A LOGISTIC REGRESSION MODEL ON THE 2 MAIN PCAs - only works with categorical Y
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(Xtrain, ytrain)

#predicting the test set results
ypred = classifier.predict(Xtest)


#how many components? Testing the variance ratio of principal components
ex_variance=np.var(pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio )
#each result is how much of the variance is explained by each component

