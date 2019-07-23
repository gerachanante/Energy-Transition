########## 0.IMPORTING & PREPARING DATA ##########
# 1.Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 2.Importing the data
data = pd.read_excel('d.ETClean.xlsx')
#data = data[data['country ISO'] == 'FRA']

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
        'LCOE Wind Onshore ($/MWh)','LCOE PV non-tracking ($/MWh)','WBG Region',
        '##random##']
yhead = ['RE in TFEC (%)', 'CO2 per Capita (tCO2)', 'Energy Intensity (MJ/$2011 PPP GDP)',
         'RE IC per Capita (W)', 'Access to electricity (%)','WBG Income Group']

# 2.Adding random values column to compare with other variables
data['##random##'] = np.random.random(size=len(data))
  
# 3.Defining the values of X and Y
X = data[Xhead]
y = data[yhead]
        
     
########## 2.VISUALIZING THE DATA ##########
# 1.Correlation Matrix Heatmap of X
f, ax = plt.subplots(figsize=(15,13))
hm = sns.heatmap(round(X.corr(),2), vmax = 1, vmin = -1, annot=False, 
                 ax=ax, cmap="coolwarm_r",fmt='.1f', linewidths=.01,
                 )
f.subplots_adjust(top=0.95)
t = f.suptitle('Energy Transition Enabling Factors Correlation Heatmap', fontsize=16)

# 2.Correlation Matrix Heatmap of Y
f, ax = plt.subplots(figsize=(5,4))
hm = sns.heatmap(round(y.corr(),2), vmax = 1, vmin = -1, annot=True, 
                 ax=ax, cmap="coolwarm_r",fmt='.2f', linewidths=.03,
                 )
f.subplots_adjust(top=0.91)
t = f.suptitle('Energy Transition Results Correlation Heatmap', fontsize=16)

# 3.Pair-wise Scatter Plots of Y
pp = sns.pairplot(y, height=2.5, aspect=1.2,hue="WBG Income Group",
                  plot_kws=dict(edgecolor="k", linewidth=0.1),
                  diag_kind="kde", diag_kws=dict(shade=True))
fig = pp.fig 
fig.subplots_adjust(top=0.9, wspace=0.1)
#t = fig.suptitle('Energy Transition Results Pairwise Plots', fontsize=14)













