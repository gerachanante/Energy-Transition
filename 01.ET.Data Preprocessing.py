########## 0.IMPORTING & PREPARING DATA ##########
# 1.Importing the libraries
import pandas as pd

# 2.Importing the data
data = pd.read_excel('d.ET from PBI.xlsx')
print(data.isnull().sum()) #shows missing values for each column

# 3.Dropping unnecessary columns
# for columns with specific names
data.drop(['ID','Non-RE IC (MW)','RE IC (MW)','Population (inhabitants)',
           'Date','TFEC (ktoe)','Electricity in TFEC (%)','RE in IC (%)',
           'Pre-existing RE IC (MW)'],
           inplace = True, axis = 1)


########## 1.IMPUTING THE MISSING DATA ##########
# 1.Creating groups for imputation
# single groups
byISO = data.groupby(['country ISO'])
byIG = data.groupby(['WBG Income Group'])
byR = data.groupby(['WBG Region'])
bysubR = data.groupby(['UN Sub-region'])
byt = data.groupby(['Year'])
# double groups 
bytIG = data.groupby(['WBG Income Group','Year'])
bytR = data.groupby(['WBG Region','Year'])
bytsubR = data.groupby(['UN Sub-region','Year'])

# 2.Country-level Imputations
# Filling up and down
data[['RE in TFEC (%)','Access to electricity (%)','Time to electricity (days)',
         'Access to clean cooking fuel & tech (%)','Renewable freshwater (bm3)']] = byISO[[
                 'RE in TFEC (%)','Access to electricity (%)','Time to electricity (days)',
                 'Access to clean cooking fuel & tech (%)',
                 'Renewable freshwater (bm3)']].fillna(
        method='ffill')
data[['Access to electricity (%)','Time to electricity (days)','Renewable freshwater (bm3)']] = byISO[[
                'Access to electricity (%)','Time to electricity (days)',
                'Renewable freshwater (bm3)']].fillna(
        method='bfill')

# Interpolation
data[['CO2 per Capita (tCO2)','Energy Intensity (MJ/$2011 PPP GDP)',
         'Access to clean cooking fuel & tech (%)','Time to start-up (days)',
         'GDP per capita, PPP ($current)','Gasoline Pump Price ($/L)','HDI',
         'PM2.5 Avg Concentration (ug/m3)',
         'Start-up Procedures Cost (% of GNI per capita)','GNI per capita, PPP ($current)',
         'Energy imports, net (% of energy use)','CapEx per IC ($/W) - Hydropower',
         'CapEx per IC ($/W) - Solid biofuels and renewable waste',
         'CapEx per IC ($/W) - Biogas','CapEx per IC ($/W) - Onshore wind energy',
         'CapEx per IC ($/W) - Solar photovoltaic','LCOE PV non-tracking ($/MWh)',
         'LCOE Wind Onshore ($/MWh)','Median Power Price ($/MWh)']] = byISO[[
                'CO2 per Capita (tCO2)','Energy Intensity (MJ/$2011 PPP GDP)',
                'Access to clean cooking fuel & tech (%)','Time to start-up (days)',
                'GDP per capita, PPP ($current)','Gasoline Pump Price ($/L)','HDI',
                'PM2.5 Avg Concentration (ug/m3)',
                'Start-up Procedures Cost (% of GNI per capita)','GNI per capita, PPP ($current)',
                'Energy imports, net (% of energy use)','CapEx per IC ($/W) - Hydropower',
                'CapEx per IC ($/W) - Solid biofuels and renewable waste',
                'CapEx per IC ($/W) - Biogas','CapEx per IC ($/W) - Onshore wind energy',
                'CapEx per IC ($/W) - Solar photovoltaic','LCOE PV non-tracking ($/MWh)',
                'LCOE Wind Onshore ($/MWh)','Median Power Price ($/MWh)']]\
         .apply(lambda i: i.interpolate(method='linear', limit_area='inside'))
# Filling up again (for some variables)
data[['CapEx per IC ($/W) - Hydropower',
         'CapEx per IC ($/W) - Solid biofuels and renewable waste',
         'CapEx per IC ($/W) - Biogas','CapEx per IC ($/W) - Onshore wind energy',
         'CapEx per IC ($/W) - Solar photovoltaic','LCOE PV non-tracking ($/MWh)',
         'LCOE Wind Onshore ($/MWh)','Median Power Price ($/MWh)']] = byISO[[
                'CapEx per IC ($/W) - Hydropower',
                'CapEx per IC ($/W) - Solid biofuels and renewable waste',
                'CapEx per IC ($/W) - Biogas','CapEx per IC ($/W) - Onshore wind energy',
                'CapEx per IC ($/W) - Solar photovoltaic','LCOE PV non-tracking ($/MWh)',
                'LCOE Wind Onshore ($/MWh)','Median Power Price ($/MWh)']].fillna(
        method='bfill')
# Extrapolation (FILLING DOWN CURRENTLY)
data[['CO2 per Capita (tCO2)','Energy Intensity (MJ/$2011 PPP GDP)',
         'Access to clean cooking fuel & tech (%)','Time to start-up (days)',
         'GDP per capita, PPP ($current)',
         'Gasoline Pump Price ($/L)','HDI',
         'PM2.5 Avg Concentration (ug/m3)','Start-up Procedures Cost (% of GNI per capita)',
         'GNI per capita, PPP ($current)','Energy imports, net (% of energy use)',
         'CapEx per IC ($/W) - Hydropower',
         'CapEx per IC ($/W) - Solid biofuels and renewable waste',
         'CapEx per IC ($/W) - Biogas','CapEx per IC ($/W) - Onshore wind energy',
         'CapEx per IC ($/W) - Solar photovoltaic','LCOE PV non-tracking ($/MWh)',
         'LCOE Wind Onshore ($/MWh)','Median Power Price ($/MWh)']] = byISO[[
                 'CO2 per Capita (tCO2)','Energy Intensity (MJ/$2011 PPP GDP)',
                 'Access to clean cooking fuel & tech (%)','Time to start-up (days)',
                 'GDP per capita, PPP ($current)',
                 'Gasoline Pump Price ($/L)','HDI',
                 'PM2.5 Avg Concentration (ug/m3)','Start-up Procedures Cost (% of GNI per capita)',
                 'GNI per capita, PPP ($current)','Energy imports, net (% of energy use)',
                 'CapEx per IC ($/W) - Hydropower',
                 'CapEx per IC ($/W) - Solid biofuels and renewable waste',
                 'CapEx per IC ($/W) - Biogas','CapEx per IC ($/W) - Onshore wind energy',
                 'CapEx per IC ($/W) - Solar photovoltaic','LCOE PV non-tracking ($/MWh)',
                 'LCOE Wind Onshore ($/MWh)','Median Power Price ($/MWh)']]\
         .apply(lambda i: i.interpolate(method='linear', limit_area='outside'))
# Median
data[['Access to electricity (%)','Time to electricity (days)','CO2 intensity (kg per kgoe energy use)',
         'Government Effectiveness (+-2.5)','Regulatory Quality (+-2.5)','Avg PV Out (kWh/kWp/y)',
         'Median Time Commitment-Commissioning (days)','Start-up Procedures Cost (% of GNI per capita)',
         'Time to start-up (days)','Energy imports, net (% of energy use)']] = byISO[[
                 'Access to electricity (%)','Time to electricity (days)',
                 'CO2 intensity (kg per kgoe energy use)','Government Effectiveness (+-2.5)',
                 'Regulatory Quality (+-2.5)','Avg PV Out (kWh/kWp/y)','Median Time Commitment-Commissioning (days)',
                 'Start-up Procedures Cost (% of GNI per capita)','Time to start-up (days)',
                 'Energy imports, net (% of energy use)']]\
    .transform(lambda i: i.fillna(i.median()))

# 3.Group-level imputation
# Median
data[['Access to electricity (%)','Time to electricity (days)','CO2 intensity (kg per kgoe energy use)',
         'Government Effectiveness (+-2.5)','Regulatory Quality (+-2.5)',
         'Median Time Commitment-Commissioning (days)',
         'Median Power Price ($/MWh)']] = byIG[[
                 'Access to electricity (%)','Time to electricity (days)',
                 'CO2 intensity (kg per kgoe energy use)',
                 'Government Effectiveness (+-2.5)','Regulatory Quality (+-2.5)',
                 'Median Time Commitment-Commissioning (days)',
                 'Median Power Price ($/MWh)']]\
    .transform(lambda i: i.fillna(i.median()))
# Yearly median
data[['CO2 per Capita (tCO2)','Energy Intensity (MJ/$2011 PPP GDP)',
         'Access to clean cooking fuel & tech (%)','Time to start-up (days)',
         'GDP per capita, PPP ($current)','Gasoline Pump Price ($/L)','HDI',
         'Start-up Procedures Cost (% of GNI per capita)',
         'GNI per capita, PPP ($current)','CapEx per IC ($/W) - Hydropower',
         'CapEx per IC ($/W) - Solid biofuels and renewable waste',
         'CapEx per IC ($/W) - Biogas','CapEx per IC ($/W) - Onshore wind energy',
         'CapEx per IC ($/W) - Solar photovoltaic']] = bytIG[[
                    'CO2 per Capita (tCO2)','Energy Intensity (MJ/$2011 PPP GDP)',
                    'Access to clean cooking fuel & tech (%)','Time to start-up (days)',
                    'GDP per capita, PPP ($current)','Gasoline Pump Price ($/L)','HDI',
                    'Start-up Procedures Cost (% of GNI per capita)',
                    'GNI per capita, PPP ($current)','CapEx per IC ($/W) - Hydropower',
                     'CapEx per IC ($/W) - Solid biofuels and renewable waste',
                     'CapEx per IC ($/W) - Biogas','CapEx per IC ($/W) - Onshore wind energy',
                     'CapEx per IC ($/W) - Solar photovoltaic']]\
    .transform(lambda i: i.fillna(i.median()))

# 4.Sub-Region-level
# Median
data[['Avg PV Out (kWh/kWp/y)','Avg Wind Power Density (W/m2)']] = bysubR[[
                 'Avg PV Out (kWh/kWp/y)','Avg Wind Power Density (W/m2)']]\
    .transform(lambda i: i.fillna(i.median()))
# Yearly median
data[['PM2.5 Avg Concentration (ug/m3)','Avg NPP (gC/m2/y)']] = bytsubR[[
                'PM2.5 Avg Concentration (ug/m3)','Avg NPP (gC/m2/y)']]\
    .transform(lambda i: i.fillna(i.median()))

# 5.Region level
# Median
data[['Avg PV Out (kWh/kWp/y)','Avg Wind Power Density (W/m2)']] = byR[[
                 'Avg PV Out (kWh/kWp/y)','Avg Wind Power Density (W/m2)']]\
    .transform(lambda i: i.fillna(i.median()))
# Yearly median
data[['PM2.5 Avg Concentration (ug/m3)','Avg NPP (gC/m2/y)']] = bytR[[
                'PM2.5 Avg Concentration (ug/m3)','Avg NPP (gC/m2/y)']]\
    .transform(lambda i: i.fillna(i.median()))

# 6.No level (All)
# Yearly Median
data[['LCOE PV non-tracking ($/MWh)','LCOE Wind Onshore ($/MWh)']] = byt[[
                 'LCOE PV non-tracking ($/MWh)','LCOE Wind Onshore ($/MWh)']]\
    .transform(lambda i: i.fillna(i.median()))
# Median
data[['Avg PV Out (kWh/kWp/y)']] = data[[
                 'Avg PV Out (kWh/kWp/y)']]\
    .transform(lambda i: i.fillna(i.median()))
# 0
data[['RE in TFEC (%)','RE IC per Capita (W)','Renewable freshwater (bm3)',
         'Energy imports, net (% of energy use)','R&D expenditure (% of GDP)',
         'RE per Auction (MW)','Committed per project ($2016M)']] = data[[
                 'RE in TFEC (%)','RE IC per Capita (W)','Renewable freshwater (bm3)',
                 'Energy imports, net (% of energy use)',
                 'R&D expenditure (% of GDP)','RE per Auction (MW)',
                 'Committed per project ($2016M)']].fillna(0)


########## 2.CONVERTING CATEGORICAL DATA TO NUMERIC ##########
# 1.Saving a categorical data version
data.to_excel(r'D:\OneDrive - International Renewable Energy Agency - IRENA\05.Master Thesis\05.Analysis\01.Working directory\d.ETClean.xlsx', 
                 sheet_name='Sheet1', index=False)

# 2.Dropping categorical columns that are not needed in analysis
data.drop(['country ISO'], inplace=True, axis=1)
data.drop(['UN Sub-region'], inplace=True, axis=1)
data.drop(['WBG Region'], inplace=True, axis=1)

# 3.Creating data sub-sets
dataHI = data[data['WBG Income Group'] == 'High income']
dataUM = data[data['WBG Income Group'] == 'Upper middle income']
dataLM = data[data['WBG Income Group'] == 'Lower middle income']
dataLI = data[data['WBG Income Group'] == 'Low income']

# 4.Transposing categories to dummy variables
dummy = pd.get_dummies(data)
dummyHI = pd.get_dummies(dataHI)
dummyUM = pd.get_dummies(dataUM)
dummyLM = pd.get_dummies(dataLM)
dummyLI = pd.get_dummies(dataLI)


########## 3.SCALING THE DATA ##########
# 1. Defining the scaling method
# MinMax Scaler will scale from 0 - 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

#2. Creating scaled dataset 
scaled = pd.DataFrame(sc.fit_transform(dummy), columns = dummy.columns)
scaledHI = pd.DataFrame(sc.fit_transform(dummyHI), columns = dummyHI.columns)
scaledUM = pd.DataFrame(sc.fit_transform(dummyUM), columns = dummyUM.columns)
scaledLM = pd.DataFrame(sc.fit_transform(dummyLM), columns = dummyLM.columns)
scaledLI = pd.DataFrame(sc.fit_transform(dummyLI), columns = dummyLI.columns)
        

########## 4.EXPORTING THE DATASETS ##########
dummy.to_excel(r'D:\OneDrive - International Renewable Energy Agency - IRENA\05.Master Thesis\05.Analysis\01.Working directory\d.ETCleanDummy.xlsx', 
               sheet_name='Sheet1', index=False)
dummyHI.to_excel(r'D:\OneDrive - International Renewable Energy Agency - IRENA\05.Master Thesis\05.Analysis\01.Working directory\d.ETCleanDummyHI.xlsx', 
                 sheet_name='Sheet1', index=False)
dummyUM.to_excel(r'D:\OneDrive - International Renewable Energy Agency - IRENA\05.Master Thesis\05.Analysis\01.Working directory\d.ETCleanDummyUM.xlsx', 
                 sheet_name='Sheet1', index=False)
dummyLM.to_excel(r'D:\OneDrive - International Renewable Energy Agency - IRENA\05.Master Thesis\05.Analysis\01.Working directory\d.ETCleanDummyLM.xlsx', 
                 sheet_name='Sheet1', index=False)
dummyLI.to_excel(r'D:\OneDrive - International Renewable Energy Agency - IRENA\05.Master Thesis\05.Analysis\01.Working directory\d.ETCleanDummyLI.xlsx', 
                 sheet_name='Sheet1', index=False)
scaled.to_excel(r'D:\OneDrive - International Renewable Energy Agency - IRENA\05.Master Thesis\05.Analysis\01.Working directory\d.ETCleanScaledDummy.xlsx', 
                sheet_name='Sheet1', index=False)
scaledHI.to_excel(r'D:\OneDrive - International Renewable Energy Agency - IRENA\05.Master Thesis\05.Analysis\01.Working directory\d.ETCleanScaledDummyHI.xlsx', 
                sheet_name='Sheet1', index=False)
scaledUM.to_excel(r'D:\OneDrive - International Renewable Energy Agency - IRENA\05.Master Thesis\05.Analysis\01.Working directory\d.ETCleanScaledDummyUM.xlsx', 
                sheet_name='Sheet1', index=False)
scaledLM.to_excel(r'D:\OneDrive - International Renewable Energy Agency - IRENA\05.Master Thesis\05.Analysis\01.Working directory\d.ETCleanScaledDummyLM.xlsx', 
                sheet_name='Sheet1', index=False)
scaledLI.to_excel(r'D:\OneDrive - International Renewable Energy Agency - IRENA\05.Master Thesis\05.Analysis\01.Working directory\d.ETCleanScaledDummyLI.xlsx', 
                sheet_name='Sheet1', index=False)