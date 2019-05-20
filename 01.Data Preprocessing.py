## 0.IMPORTING & EXPLORATORY ANALYSIS
# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the data
dataset = pd.read_excel('d.ET from PBI.xlsx')
print(dataset.isnull().sum()) #shows missing values for each column
dataset.info() #basic info for each column

#dropping columns
"""
dataset.drop([dataset.columns[i] for i in [1, 2, 3]], inplace=True, axis=1)
"""
#new conditions
"""
# Three level nesting with np.where
np.where(if_this_condition_is_true_one, do_this, 
  np.where(if_this_condition_is_true_two, do_that, 
    np.where(if_this_condition_is_true_three, do_foo, do_bar)))
"""

## 1.IMPUTING THE MISSING DATA
#Dropping missing data
"""
dataset.dropna() #rows
dataset.dropna(axis = 1) #columns
dataset.dropna(thresh=int(dataset.shape[0]*.9),axis=1) #columns with at least 90% valid
"""

# Create the group and time objects
"""bygroup = dataset.groupby(['columnname'])"""
byISO = dataset.groupby(['country ISO'])
byIG = dataset.groupby(['WBG Income Group'])
bytIG = dataset.groupby(['WBG Income Group','Year'])
bytR = dataset.groupby(['UN Sub-Region','Year'])

""" #Methods to imputate below
dataset[['col1','col2']] = bygroup[['col1','col2']].fillna(method=
'ffill' or 'bfill', limit = #)
dataset[['col1','col2']] = bygroup[['col1','col2']]\
.apply(lambda i: i.interpolate(method = 'linear', limit_area = 'inside'))
.transform(lambda i: i.fillna(i.median()))
"""

#Country-level
#Filling up and down
dataset[['RE in TFEC (%)','Access to electricity (%)','Time to electricity (days)',
         'Access to clean cooking fuel & tech (%)','RE share in electricity (%)',
         'Renewable freshwater (bm3)']] = byISO[[
                 'RE in TFEC (%)','Access to electricity (%)','Time to electricity (days)',
                 'Access to clean cooking fuel & tech (%)','RE share in electricity (%)',
                 'Renewable freshwater (bm3)']].fillna(
        method='ffill')
dataset[['Access to electricity (%)','Time to electricity (days)','Renewable freshwater (bm3)']] = byISO[[
                'Access to electricity (%)','Time to electricity (days)',
                'Renewable freshwater (bm3)']].fillna(
        method='bfill')
dataset[['Pre-existing RE IC (MW)']] = byISO[['Pre-existing RE IC (MW)']].fillna(
        method='bfill', limit=1)
#Interpolation
dataset[['GHG per capita (tCO2)','Energy Intensity (MJ/$2011 PPP GDP)',
         'Access to clean cooking fuel & tech (%)','Time to start-up (days)',
         'RE share in electricity (%)','GDP per capita, PPP ($current)',
         'Gasoline Pump Price ($/L)','HDI','Energy use per capita (kgoe)',
         'PM2.5 Avg Concentration (ug/m3)','Start-up Procedures Cost (% of GNI per capita)',
         'GNI per capita, PPP ($current)','Energy imports, net (% of energy use)']] = byISO[[
                 'GHG per capita (tCO2)','Energy Intensity (MJ/$2011 PPP GDP)',
                 'Access to clean cooking fuel & tech (%)','Time to start-up (days)',
                 'RE share in electricity (%)','GDP per capita, PPP ($current)',
                 'Gasoline Pump Price ($/L)','HDI','Energy use per capita (kgoe)',
                 'PM2.5 Avg Concentration (ug/m3)','Start-up Procedures Cost (% of GNI per capita)',
                 'GNI per capita, PPP ($current)','Energy imports, net (% of energy use)']]\
         .apply(lambda i: i.interpolate(method='linear', limit_area='inside'))
#Extrapolation (FILLING DOWN CURRENTLY)
dataset[['GHG per capita (tCO2)','Energy Intensity (MJ/$2011 PPP GDP)',
         'Access to clean cooking fuel & tech (%)','Time to start-up (days)',
         'RE share in electricity (%)','GDP per capita, PPP ($current)',
         'Gasoline Pump Price ($/L)','HDI','Energy use per capita (kgoe)',
         'PM2.5 Avg Concentration (ug/m3)','Start-up Procedures Cost (% of GNI per capita)',
         'GNI per capita, PPP ($current)','Energy imports, net (% of energy use)']] = byISO[[
                 'GHG per capita (tCO2)','Energy Intensity (MJ/$2011 PPP GDP)',
                 'Access to clean cooking fuel & tech (%)','Time to start-up (days)',
                 'RE share in electricity (%)','GDP per capita, PPP ($current)',
                 'Gasoline Pump Price ($/L)','HDI','Energy use per capita (kgoe)',
                 'PM2.5 Avg Concentration (ug/m3)','Start-up Procedures Cost (% of GNI per capita)',
                 'GNI per capita, PPP ($current)','Energy imports, net (% of energy use)']]\
         .apply(lambda i: i.interpolate(method='linear', limit_area='outside'))
#Median
dataset[['Access to electricity (%)','Time to electricity (days)','CO2 intensity (kg per kgoe energy use)',
         'Government Effectiveness (+-2.5)','Regulatory Quality (+-2.5)',
         'Median Time Committment-Commissioning (days)','Start-up Procedures Cost (% of GNI per capita)',
         'Time to start-up (days)','Energy imports, net (% of energy use)']] = byISO[[
                 'Access to electricity (%)','Time to electricity (days)',
                 'CO2 intensity (kg per kgoe energy use)','Government Effectiveness (+-2.5)',
                 'Regulatory Quality (+-2.5)','Median Time Committment-Commissioning (days)',
                 'Start-up Procedures Cost (% of GNI per capita)','Time to start-up (days)',
                 'Energy imports, net (% of energy use)']]\
    .transform(lambda i: i.fillna(i.median()))
#Group-level
#Median
dataset[['Access to electricity (%)','Time to electricity (days)','CO2 intensity (kg per kgoe energy use)',
         'Government Effectiveness (+-2.5)','Regulatory Quality (+-2.5)',
         'Median Time Committment-Commissioning (days)']] = byIG[[
                 'Access to electricity (%)','Time to electricity (days)',
                 'CO2 intensity (kg per kgoe energy use)',
                 'Government Effectiveness (+-2.5)','Regulatory Quality (+-2.5)',
                 'Median Time Committment-Commissioning (days)']]\
    .transform(lambda i: i.fillna(i.median()))
#Yearly median
dataset[['GHG per capita (tCO2)','Energy Intensity (MJ/$2011 PPP GDP)',
         'Access to clean cooking fuel & tech (%)','Time to start-up (days)',
         'GDP per capita, PPP ($current)','Gasoline Pump Price ($/L)','HDI',
         'Energy use per capita (kgoe)','Start-up Procedures Cost (% of GNI per capita)',
         'GNI per capita, PPP ($current)']] = bytIG[[
                    'GHG per capita (tCO2)','Energy Intensity (MJ/$2011 PPP GDP)',
                    'Access to clean cooking fuel & tech (%)','Time to start-up (days)',
                    'GDP per capita, PPP ($current)','Gasoline Pump Price ($/L)','HDI',
                    'Energy use per capita (kgoe)','Start-up Procedures Cost (% of GNI per capita)',
                    'GNI per capita, PPP ($current)']]\
    .transform(lambda i: i.fillna(i.median()))
#Region-level
#Yearly median
dataset[['PM2.5 Avg Concentration (ug/m3)']] = bytR[['PM2.5 Avg Concentration (ug/m3)']]\
    .transform(lambda i: i.fillna(i.median()))
#No level (All)
#0
dataset[['RE in TFEC (%)','RE IC per Capita (W/person)','Renewable freshwater (bm3)',
         'Pre-existing RE IC (MW)','Energy imports, net (% of energy use)',
         'RE share in electricity (%)','R&D expenditure (% of GDP)',
         'Committed per project ($2016M)','RE in IC (%)']] = dataset[[
                 'RE in TFEC (%)','RE IC per Capita (W/person)','Renewable freshwater (bm3)',
                 'Pre-existing RE IC (MW)','Energy imports, net (% of energy use)',
                 'RE share in electricity (%)','R&D expenditure (% of GDP)',
                 'Committed per project ($2016M)','RE in IC (%)']].fillna(0)

## 2.CATEGORICAL DATA
#Changing categorical to numeric (text)
"""
from sklearn.preprocessing import LabelEncoder
for i in dataset:
    if dataset[i].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(dataset[i].values))
        dataset[i] = lbl.transform(list(dataset[i].values))
"""

#dropping country ISO before dummy
"""dataset.drop(['country ISO'], inplace=True, axis=1)
"""
#Transposing categories to dummy variables
dummyset = pd.get_dummies(dataset) #LabelEncoder not needed if using this

## 3.SCALING THE FEATURES
from sklearn.preprocessing import MinMaxScaler #minmax scales on a range 0 - 1
sc = MinMaxScaler()
#new variable called scaledset for exporting
scaledset = pd.DataFrame(sc.fit_transform(dummyset), columns = dummyset.columns)
        
## 4.EXPORTING DATASET TO XLSX
dataset.to_excel(r'D:\OneDrive - International Renewable Energy Agency - IRENA\05.Master Thesis\05.Analysis\01.Python scripts\d.ETClean.xlsx', 
                 sheet_name='Sheet1', index=False)
dummyset.to_excel(r'D:\OneDrive - International Renewable Energy Agency - IRENA\05.Master Thesis\05.Analysis\01.Python scripts\d.ETCleanDummy.xlsx', 
                 sheet_name='Sheet1', index=False)
scaledset.to_excel(r'D:\OneDrive - International Renewable Energy Agency - IRENA\05.Master Thesis\05.Analysis\01.Python scripts\d.ETCleanScaledDummy.xlsx', 
                 sheet_name='Sheet1', index=False)


















