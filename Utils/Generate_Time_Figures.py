# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:14:35 2022

@author: jpeeples
"""
import matplotlib.pyplot as plt
import numpy as np

def compute_computational_cost(time_dictionary,features,folder,num_SP,num_runs):
    
    
    #For each feature, plot number of bins vs computational time
    fig, ax = plt.subplots()
    
    #Add leading zero for number of bins
    num_SP = np.append(0,num_SP)
    
    for feature in features:
        
        temp_matrix = time_dictionary[feature]
        meanst = np.array(np.append(0,np.mean(temp_matrix,axis=0)), dtype=np.float64)
        sdt = np.array(np.append(0,np.std(temp_matrix,axis=0)), dtype=np.float64)
        ax.plot(num_SP, meanst, label=feature)
        ax.fill_between(num_SP, meanst-sdt, meanst+sdt ,alpha=0.3)
    
    ax.legend()
    
    plt.title('Average Computational Cost for Runs {} and {}'.format(num_runs[0],
                                                                     num_runs[1]),
              fontsize=16)
    plt.xlabel('Number of Bins',fontsize=14)
    plt.ylabel('Computational Time (s)', fontsize=14)
    plt.xticks(rotation = 45)
    ax.margins(x=0)
    ax.set_xticks(num_SP)
    plt.savefig('{}Compuational Time for Runs {}'.format(folder,num_runs))
    plt.tight_layout()
    plt.close(fig)
        