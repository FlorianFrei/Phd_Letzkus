# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:24:50 2024

@author: Florian Freitag
"""

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import numpy as np
import pandas as pd
#from Zeta_helper import*
import itertools
import matplotlib.pyplot as plt

import zetapy as zp

#%%

Data = pd.read_csv("E:/Florian_paper/Florian/Aligned_Data/otpo/wrangled_Data/Zetadata.csv")

#%% for single tries

# =============================================================================
# Data3 = Data.loc[(Data['Day'] == 1) & (Data['Mouse'] == "M7")]
#  
# timess = np.asarray(Data3.loc[Data3[Data3.Trialtype=="Airpuff"].groupby('Trial')['time_bin'].idxmin()]['time_bin'])[np.newaxis].T-0.5
# Spikes = np.asarray(Data3.query('cluster_id==101').loc[:,'time_bin'])[np.newaxis].T
# zp.zetatest(Spikes, timess,boolPlot=True)
#  #%% for single triess
# Data3 = Data.loc[(Data['Day'] == 1) & (Data['Mouse'] == "M7")]
# Zetatest_unit(Data3.loc[Data3['Behv']=='PC'], 'Airpuff2', 176,window=2,lag=0.1,Plot=True)
# Zetatest_unit(Data3.loc[Data3['Behv']=='A2L'], 'Airpuff', 176,window=2,lag=0.1,Plot=True)
# Zetatest_unit(Data3.loc[Data3['Behv']=='W2T'], 'HIT', 176,window=2,lag=0.5,Plot=True)
# Zetatest_unit(Data3.loc[Data3['Behv']=='W2T'], 'Audio', 176,window=2,lag=0.5,Plot=True)
# Zetatest_unit(Data3.loc[Data3['Behv']=='W2T'], 'LeftReward', 176,window=2,lag=0.5,Plot=True)
# =============================================================================
#     
# =============================================================================
#%% Zeta_all
W2T = ZetaMass(Data,'W2T','HIT',lag = 0.01, window = 0.3)
A2L = ZetaMass(Data,'A2L','Airpuff',lag=0.01, window = 0.3)
PC = ZetaMass(Data,'PC','Airpuff2',lag=0.01, window = 0.3)
Whisk = ZetaMass(Data,'W2T','HIT',lag=0.3, window = 0.3)
Lick_W2T = ZetaMass(Data,'W2T','LeftReward',lag=0, window = 0.3)
Lick_A2L = ZetaMass(Data,'A2L','HIT',lag=0.05, window = 0.3)

#%%
basefolder = "E:/Florian/Aligned_Data/Analysable_data/wrangled_Data/Zetatests"
W2T.to_csv(basefolder + str('/W2T.csv'))
A2L.to_csv(basefolder + str('/A2L.csv'))
PC.to_csv(basefolder + str('/PC.csv'))
Whisk.to_csv(basefolder + str('/Whisk.csv'))
Lick_W2T.to_csv(basefolder + str('/lickW2T.csv'))
Lick_A2L.to_csv(basefolder + str('/lickA2L.csv'))


#%% Zeta opto
pd.unique(Data['Behv'])
pd.unique(Data['Trialtype'])

Airpuff = ZetaMass(Data,'Airpuff','AirTop_noOpto',lag = 0, window = 0.2)
Opto = ZetaMass(Data,'just_opto','justOpto',lag = 0, window = 0.2)
Opto_Air =  ZetaMass(Data,'Opto_Air','AirTop_Opto',lag = 0, window = 0.2)

#%%
basefolder = "E:/Florian_paper/Florian/Aligned_Data/otpo/wrangled_Data/Zetatests"
Opto.to_csv(basefolder + str('/Opto.csv'))
Airpuff.to_csv(basefolder + str('/Airpuff.csv'))
Opto_Air.to_csv(basefolder + str('/Opto_Air.csv'))



#%%



u_list = []
P_list = []
for unit in Ephys_good['cluster_id'].unique():
    u_list.append(unit)
    Spikes = Ephys_good[Ephys_good['cluster_id']==unit]
    dblZetaP = zp.zetatest(np.array(Spikes['seconds']), np.array(ttlsound),dblUseMaxDur=5,boolPlot=False,intResampNum = 1000)[0] 
    P_list.append(dblZetaP)

#%%

zeta = pd.DataFrame({'unit':u_list, 'p_val':P_list})
zeta.to_csv('D:/3556-17/3556-17_naive_g0/Zeta_sound_All.csv',index=False)


#%%
#e = zp.zetatest(np.array(Ephys_good[Ephys_good['cluster_id']==130]['seconds']), np.array(ttlsound),dblUseMaxDur=6,boolPlot=True,intResampNum = 1000)
ul = Ephys_good['cluster_id'].unique()
e = zp.zetatest(np.array(Ephys_good[Ephys_good['cluster_id']==ul[8]]['seconds']), np.array(ttlsound-1),dblUseMaxDur=7,boolPlot=True,intResampNum = 100)