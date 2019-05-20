## Random Forest Regression

## 1. PREPARING THE DATA
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the data
dataset = pd.read_excel('d.ETClean.xlsx')

## 2.DEFINING X & Y
Xhead = ['HDI','Government Effectiveness (+-2.5)','Regulatory Quality (+-2.5)',
        'Time to electricity (days)','RE share in electricity (%)',
        'GDP per capita, PPP ($current)','Renewable freshwater (bm3)',
        'Energy use per capita (kgoe)','Non-RE Net Decom. 2000 (MW)',
        'Median Time Committment-Commissioning (days)','Access to clean cooking fuel & tech (%)',
        'RE in IC (%)','Pre-existing RE IC (MW)','Strategic planning','Economic Instruments',
        'Policy Support','Technology deployment and diffusion','Regulatory Instrument',
        'Loans','Information provision','Information and Education','Voluntary Approaches',
        'Grants and subsidies','GHG emissions trading','Comparison label','Obligation schemes',
        'Building codes and standards','Energy imports, net (% of energy use)',
        'CO2 intensity (kg per kgoe energy use)','Policies','FIT/Premiums',
        'Performance label','Technology deployment','RD&D',
        'PM2.5 Avg Concentration (ug/m3)','Gasoline Pump Price ($/L)',
        'Start-up Procedures Cost (% of GNI per capita)','Time to start-up (days)',
        'GNI per capita, PPP ($current)','R&D expenditure (% of GDP)',
        'Committed per project ($2016M)','Year','WBG Income Group_High income',
        'WBG Income Group_Low income','WBG Income Group_Lower middle income',
        'WBG Income Group_Upper middle income','UN Sub-Region_Australia and New Zealand',
        'UN Sub-Region_Central Asia','UN Sub-Region_Eastern Asia','UN Sub-Region_Eastern Europe',
        'UN Sub-Region_Latin America and the Caribbean','UN Sub-Region_Melanesia',
        'UN Sub-Region_Micronesia','UN Sub-Region_Northern Africa','UN Sub-Region_Northern America',
        'UN Sub-Region_Northern Europe','UN Sub-Region_Polynesia','UN Sub-Region_South-eastern Asia',
        'UN Sub-Region_Southern Asia','UN Sub-Region_Southern Europe','UN Sub-Region_Sub-Saharan Africa',
        'UN Sub-Region_Western Asia','UN Sub-Region_Western Europe','country ISO_ZWE',
        'country ISO_ZMB','country ISO_ZAF','country ISO_YEM','country ISO_XKX',
        'country ISO_WSM','country ISO_VUT','country ISO_VNM','country ISO_VIR',
        'country ISO_VGB','country ISO_VEN','country ISO_VCT','country ISO_UZB',
        'country ISO_USA','country ISO_URY','country ISO_UKR','country ISO_UGA',
        'country ISO_TZA','country ISO_TUV','country ISO_TUR','country ISO_TUN',
        'country ISO_TTO','country ISO_TON','country ISO_TLS','country ISO_TKM',
        'country ISO_TJK','country ISO_THA','country ISO_TGO','country ISO_TCD',
        'country ISO_TCA','country ISO_SYR','country ISO_SYC','country ISO_SWZ',
        'country ISO_SWE','country ISO_SVN','country ISO_SVK','country ISO_SUR',
        'country ISO_STP','country ISO_SSD','country ISO_SRB','country ISO_SOM',
        'country ISO_SLV','country ISO_SLE','country ISO_SLB','country ISO_SGP',
        'country ISO_SEN','country ISO_SDN','country ISO_SAU','country ISO_RWA',
        'country ISO_RUS','country ISO_ROU','country ISO_QAT','country ISO_PYF',
        'country ISO_PSE','country ISO_PRY','country ISO_PRT','country ISO_PRK',
        'country ISO_PRI','country ISO_POL','country ISO_PNG','country ISO_PLW',
        'country ISO_PHL','country ISO_PER','country ISO_PAN','country ISO_PAK',
        'country ISO_OMN','country ISO_NZL','country ISO_NRU','country ISO_NPL',
        'country ISO_NOR','country ISO_NLD','country ISO_NIC','country ISO_NGA',
        'country ISO_NER','country ISO_NCL','country ISO_NAM','country ISO_MYS',
        'country ISO_MWI','country ISO_MUS','country ISO_MTQ','country ISO_MRT',
        'country ISO_MOZ','country ISO_MNG','country ISO_MNE','country ISO_MMR',
        'country ISO_MLT','country ISO_MLI','country ISO_MKD','country ISO_MHL',
        'country ISO_MEX','country ISO_MDV','country ISO_MDG','country ISO_MDA',
        'country ISO_MAR','country ISO_MAF','country ISO_LVA','country ISO_LUX',
        'country ISO_LTU','country ISO_LSO','country ISO_LKA','country ISO_LIE',
        'country ISO_LCA','country ISO_LBY','country ISO_LBR','country ISO_LBN',
        'country ISO_LAO','country ISO_KWT','country ISO_KOR','country ISO_KNA',
        'country ISO_KIR','country ISO_KHM','country ISO_KGZ','country ISO_KEN',
        'country ISO_KAZ','country ISO_JPN','country ISO_JOR','country ISO_JAM',
        'country ISO_ITA','country ISO_ISR','country ISO_ISL','country ISO_IRQ',
        'country ISO_IRN','country ISO_IRL','country ISO_IND','country ISO_IDN',
        'country ISO_HUN','country ISO_HTI','country ISO_HRV','country ISO_HND',
        'country ISO_GUY','country ISO_GUM','country ISO_GTM','country ISO_GRL',
        'country ISO_GRD','country ISO_GRC','country ISO_GNQ','country ISO_GNB',
        'country ISO_GMB','country ISO_GIN','country ISO_GHA','country ISO_GEO',
        'country ISO_GBR','country ISO_GAB','country ISO_FSM','country ISO_FRO',
        'country ISO_FRA','country ISO_FJI','country ISO_FIN','country ISO_ETH',
        'country ISO_EST','country ISO_ESP','country ISO_ERI','country ISO_EGY',
        'country ISO_ECU','country ISO_DZA','country ISO_DOM','country ISO_DNK',
        'country ISO_DMA','country ISO_DJI','country ISO_DEU','country ISO_CZE',
        'country ISO_CYP','country ISO_CYM','country ISO_CUW','country ISO_CUB',
        'country ISO_CRI','country ISO_CPV','country ISO_COM','country ISO_COL',
        'country ISO_COG','country ISO_COD','country ISO_CMR','country ISO_CIV',
        'country ISO_CHN','country ISO_CHL','country ISO_CHE','country ISO_CAN',
        'country ISO_CAF','country ISO_BWA','country ISO_BTN','country ISO_BRN',
        'country ISO_BRB','country ISO_BRA','country ISO_BOL','country ISO_BLZ',
        'country ISO_BLR','country ISO_BIH','country ISO_BHS','country ISO_BHR',
        'country ISO_BGR','country ISO_BGD','country ISO_BFA','country ISO_BEN',
        'country ISO_BEL','country ISO_BDI','country ISO_AZE','country ISO_AUT',
        'country ISO_AUS','country ISO_ATG','country ISO_ASM','country ISO_ARM',
        'country ISO_ARG','country ISO_ARE','country ISO_ALB','country ISO_AGO',
        'country ISO_AFG','country ISO_ABW']
yhead = ['RE in TFEC (%)', 'GHG per capita (tCO2)', 'Energy Intensity (MJ/$2011 PPP GDP)',
           'RE IC per Capita (W/person)', 'Access to electricity (%)']
X = dataset[Xhead] #includes all Xs, numerical and categorical
y = dataset[yhead] #all the dependent variables y (6 variables)

## 3. SPLITTING THE DATA FOR TRAINING A MACHINE LEARNING MODEL
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## 4. RANDOM FOREST REGRESSION
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 0, 
                           max_depth=None, max_features = "auto",
                           oob_score = True, )
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

#Testing the random forest regression's accuracy
rf.score(X_test,y_test)

## 5. RANDOM FOREST RESULTS
#Viewing the importance of the Xs
print(rf.feature_importances_)
print(rf.oob_score_)

# E. Tuning the Hyperparameters of the Random Forest
"""start = time.time()

param_dist = {'n_estimators':[500],
              'max_depth': [2, 3, 4],
              'bootstrap': [True, False],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'criterion': ['gini', 'entropy']}

cv_rf = GridSearchCV(fit_rf, cv = 10,
                     param_grid=param_dist, 
                     n_jobs = 3)

cv_rf.fit(training_set, class_set)
print('Best Parameters using grid search: \n', 
      cv_rf.best_params_)
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))
"""

#OTHER CODE THAT COULD BE USEFUL
# Visualizing the prediction matrix
plt.figure(figsize=(10,10))
sns.heatmap(rf, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


#dealing more specifically with multioutput
"""
class MultiOutputRF(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
    def fit(self, X, Y):
        X, Y = map(np.atleast_2d, (X, Y))
        assert X.shape[0] == Y.shape[0]
        Ny = Y.shape[1]
        
        self.clfs = []
        for i in range(Ny):
            clf = RandomForestRegressor(*self.args, **self.kwargs)
            Xi = np.hstack([X, Y[:, :i]])
            yi = Y[:, i]
            self.clfs.append(clf.fit(Xi, yi))
            
        return self
        
    def predict(self, X):
        Y = np.empty([X.shape[0], len(self.clfs)])
        for i, clf in enumerate(self.clfs):
            Y[:, i] = clf.predict(np.hstack([X, Y[:, :i]]))
        return Y

rf2 = MultiOutputRF(100).fit(X_train, y_train)
y_pred2 = rf2.predict(X_test)

#graph
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    
ax[0].plot(y_pred[:, 0], y_pred[:, 1], 'o', alpha=0.7)
ax[1].plot(y_pred2[:, 0], y_pred2[:, 1], 'o', alpha=0.7)

ax[0].set_title("Standard method")
ax[1].set_title("Daisy-chain method")

for axi in ax:
    axi.add_patch(plt.Rectangle((6, 6), 14, 14, color='yellow', alpha=0.2))
    axi.set_xlim(2, 12)
    axi.set_ylim(2, 12)
"""
