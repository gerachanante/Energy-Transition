# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:25:58 2019

@author: GEscamilla
"""

## 0.IMPORTING & EXPLORATORY ANALYSIS
# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the data
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
        
## 4.VISUALIZING THE DATASET
# Correlation Matrix Heatmap
f, ax = plt.subplots(figsize=(5,4))
hm = sns.heatmap(round(y.corr(),2), annot=True, ax=ax, cmap="coolwarm",fmt='.1f',
                 linewidths=.02)
f.subplots_adjust(top=0.86)
t= f.suptitle('Energy Transition Correlation Heatmap', fontsize=16)
# Pair-wise Scatter Plots
pp = sns.pairplot(y, height=2.8, aspect=1.1,
                  plot_kws=dict(edgecolor="k", linewidth=0.3),
                  diag_kind="kde", diag_kws=dict(shade=True))
fig = pp.fig 
fig.subplots_adjust(top=0.95, wspace=0.1)
t = fig.suptitle('Energy Transition Results Pairwise Plots', fontsize=14)

















