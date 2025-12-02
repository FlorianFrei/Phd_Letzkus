# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:24:50 2024

@author: Florian Freitag
"""




import os 
import numpy as np
import pandas as pd
from pathlib import Path  


import zetapy as zp


def Ephys_wrangle(cluster_info,clust,times,sample_freq):
    #takes raw KS vectos and turns them into a Dataframe  
    #selects only clusters that have the PHY good 
    
    good_cluster = cluster_info.query('bc_unitType == "MUA" |bc_unitType == "GOOD"').loc[:,['cluster_id']]
    Ephys_raw = pd.DataFrame({'cluster_id': clust, 'times':times}, columns=['cluster_id', 'times'])
    Ephys_raw = Ephys_raw.assign(seconds = Ephys_raw['times']/sample_freq)
    Ephys_good = Ephys_raw.merge(good_cluster, on=['cluster_id'],how='inner')
    return Ephys_good



#%%
basefolder ="D:/Data/3198-52/3198-52_recall_g0"
#%%

mat_name = next(file for file in os.listdir(basefolder) if file.endswith('.mat'))

imec_dir = next(d for d in Path(basefolder).iterdir() if d.is_dir() and d.name.endswith("_imec0"))
meta_path = str(next(imec_dir.glob("*.ap.meta")))


cluster_info = pd.read_csv(basefolder + '/sorted/phy/cluster_info.tsv', sep='\t')

# Load spike data
clust = np.array(np.load(basefolder + '/sorted/phy/spike_clusters.npy'))
times = np.array(np.load(basefolder + '/sorted/phy/spike_times.npy'))[:, 0]


# Load sound TTL data
ttlsound = pd.read_csv(basefolder + '/Meta/Audio.csv')
ttlsound = ttlsound[ttlsound['edge_type']=='rising'].iloc[:,0]

# Get sampling frequency
sampling_freq = float([line.split('=')[1] for line in open(meta_path) 
                      if line.startswith('imSampRate=')][0])







Ephys_good = Ephys_wrangle(cluster_info,clust,times,sampling_freq)


u_list = []
P_list = []
for unit in Ephys_good['cluster_id'].unique():
    u_list.append(unit)
    Spikes = Ephys_good[Ephys_good['cluster_id']==unit]
    dblZetaP = zp.zetatest(np.array(Spikes['seconds']), np.array(ttlsound),dblUseMaxDur=5,boolPlot=False,intResampNum = 1000)[0] 
    P_list.append(dblZetaP)


zeta = pd.DataFrame({'unit':u_list, 'p_val':P_list})
zeta.to_csv(basefolder +str("/Zeta_sound_allUnits.csv"),index=False)


#%%


# Loop through all nidq.bin files recursively
for p in Path("D:/Data/raw").rglob("*.nidq.bin"):
    basefolder = p.parent
    print(basefolder)
    
    
    output_file = basefolder / "Zeta_sound_allUnits.csv"

    if output_file.exists():
        print(f"Skipping {basefolder.name}, result already exists.")
        continue

    print(f"Processing {basefolder} ...")

    try:
        # Find .mat file
        mat_name = next(file for file in os.listdir(basefolder) if file.endswith(".mat"))

        # Find _imec0 folder and corresponding .ap.meta file
        imec_dir = next(d for d in basefolder.iterdir() if d.is_dir() and d.name.endswith("_imec0"))
        meta_path = str(next(imec_dir.glob("*.ap.meta")))

        # Load sorted spike data
        cluster_info = pd.read_csv(basefolder / "sorted/phy/cluster_info.tsv", sep="\t")
        clust = np.array(np.load(basefolder / "sorted/phy/spike_clusters.npy"))
        times = np.array(np.load(basefolder / "sorted/phy/spike_times.npy"))[:, 0]

        # Load sound TTLs
        try:
            ttlsound = pd.read_csv(basefolder / "Meta/Audio.csv")
            ttlsound = ttlsound[ttlsound["edge_type"] == "rising"].iloc[:, 0]
        except FileNotFoundError:
            ttlsound =  pd.read_csv(basefolder / 'Meta/soundttl.csv').iloc[:,0]
            
        sampling_freq = float([
            line.split("=")[1]
            for line in open(meta_path)
            if line.startswith("imSampRate=")
        ][0])

        # Run wrangling and zeta analysis
        Ephys_good = Ephys_wrangle(cluster_info, clust, times, sampling_freq)

        u_list, P_list = [], []
        for unit in Ephys_good["cluster_id"].unique():
            Spikes = Ephys_good[Ephys_good["cluster_id"] == unit]
            dblZetaP = zp.zetatest(
                np.array(Spikes["seconds"]),
                np.array(ttlsound),
                dblUseMaxDur=5,
                boolPlot=False,
                intResampNum=1000
            )[0]
            u_list.append(unit)
            P_list.append(dblZetaP)

        zeta = pd.DataFrame({"unit": u_list, "p_val": P_list})
        zeta.to_csv(output_file, index=False)
        print(f"Saved {output_file}")

    except Exception as e:
        print(f"Error processing {basefolder}: {e}")



#%%
#e = zp.zetatest(np.array(Ephys_good[Ephys_good['cluster_id']==130]['seconds']), np.array(ttlsound),dblUseMaxDur=6,boolPlot=True,intResampNum = 1000)
ul = Ephys_good['cluster_id'].unique()
e = zp.zetatest(np.array(Ephys_good[Ephys_good['cluster_id']==ul[1]]['seconds']), np.array(ttlsound-1),dblUseMaxDur=7,boolPlot=True,intResampNum = 100)