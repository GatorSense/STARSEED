# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:18:25 2020
Functions to visualize results of clustering
@author: weihuang.xu
"""

import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
import matplotlib.cm as colormap
from matplotlib import offsetbox
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import itertools
from skimage.transform import resize
from Utils.Compute_fractal_dim import fractal_dimension
import os
import pdb

#Define function to plot confusion matrix, input variables y_true
# and y_pred should be ndarray. Label is generated from y_true.
def plot_confusion_matrix(y_true, y_pred, title=None,folder='Results/'):

    
    confmatrix = confusion_matrix(y_true, y_pred, sample_weight=None,normalize=None)
    labels = np.sort(np.unique(y_true))
    df_confmatrix = pd.DataFrame(confmatrix, index = labels, columns = labels)
    accuracy = (y_true == y_pred).sum()/len(y_true)
    plot_confusion_matrix_blue(confmatrix,labels,title=title,show_percent=True)
    plt.savefig(folder+title+'_Best_CM.png')
    plt.close()
    
    print('Accuracy is %0.4f' %(accuracy))
    
def get_accuracy(y_true,y_pred):
    accuracy = (y_true == y_pred).sum()/len(y_true)
    return accuracy

def annot_max(x,y, ax=None,SP=False):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    if SP:
        text= "SP={:.3f}, Acc={:.3f}".format(xmax, ymax)
    else:
        text= "K={:.3f}, Acc={:.3f}".format(xmax, ymax)
        
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
    
def annot_max_index(x,y, ax=None,index='SI'):
    xmax = int(x[np.argmax(y)])
    ymax = y.max()
  
    text= "SP={:.3f}, {}={:.3f}".format(xmax, index, ymax)
        
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

def get_SP_plot_SI(scores,num_SP,title_type='Cultivar',folder='Results/',metric='Silhouette'):
    
    #Generate figure
    fig, ax = plt.subplots()
    ax.plot(np.array(num_SP),np.array(scores))
    
    #Plot Accuracy for each value of K and annotate max value
    if metric == 'Silhouette':
        index = 'SI'
    elif metric == 'Calinski-Harabasz':
        index = 'CHI'
    else: #Scatter metric
        index = 'SI'
    annot_max_index(np.array(num_SP),np.array(scores),ax=ax, index = index)

    if title_type is not None:
        plt.title('{} {} Index for {} values of Superpixels(SP)'.format(title_type,metric,len(num_SP)))
    else:
        plt.title('{} Index for {} values of Superpixels(SP)'.format(metric,len(num_SP)))
   
    plt.xlabel('Number of Super Pixels Considered (SP)') 
    plt.ylabel('{} Index ({})'.format(metric,index))
    
    #Save figure
    fig.savefig((folder+title_type+'_EMD_{}.png'.format(index)),dpi=fig.dpi)
    plt.close(fig=fig)
    

def get_Cluster_metrics_plots(scores,num_SP,title_type='Affinity Propagation',
                              folder='Results',
                              measures=['Homogeneity','Completeness','V-Measure'],
                              adjusted=True):
    colors = ['r','g','b']
    short_name = ['HG','CS','VM']
    if adjusted:
        measures = ['Precision','Recall','F1-Score']
        short_name = ['Precision','Recall','F1']
    #Generate figure
    fig, ax = plt.subplots(len(scores[0].keys()),len(measures),figsize=(15,8))
    
    #Get list of scores for each treatment where each element is an array for
    #plotting
    treatments = scores[0].keys()
    count = 0
    for treatment in treatments:
        indices = np.zeros((len(scores),len(measures)))
        for row in range(0,len(scores)):
            indices[row,:] = np.asarray(scores[row][treatment])
        for index in range(0,len(measures)):
            ax[count,index].plot(np.array(num_SP),indices[:,index],
                                         colors[index])
            if count == 0:
                ax[count,index].set_title(measures[index]+' Score')
            if index == 0:
                ax[count,index].set_ylabel(treatment)
            annot_max_index(np.array(num_SP),indices[:,index],
                            ax=ax[count,index],index=short_name[index])
        count += 1
    
    plt.setp(ax[-1, :], xlabel='Number of Superpixels (SP)')

    if title_type is not None:
        plt.suptitle('{} Cluster Indices for {} values of Superpixels (SP)'.format(title_type,len(num_SP)))
    else:
        plt.suptitle('Cluster Indices for {} values of Superpixels (SP)'.format(len(num_SP)))
   
    
    #Save figure
    fig.savefig((folder+title_type+'_Cluster_Indices.png'),dpi=fig.dpi)
    plt.close(fig=fig)
    
def get_SP_plot(accuracies,num_SP,title_type=None,folder='Results/'):
    
    #Generate figure
    fig, ax = plt.subplots()
    ax.plot(np.array(num_SP),np.array(accuracies))
    
    #Plot Accuracy for each value of K and annotate max value
    annot_max(np.array(num_SP),np.array(accuracies),ax=ax,SP=True)

    if title_type is not None:
        plt.title('{} Accuracy for {} values of Superpixels(SP)'.format(title_type,len(num_SP)))
    else:
        plt.title('Accuracy for {} values of Superpixels(SP)'.format(len(num_SP)))
   
    plt.xlabel('Number of Super Pixels Considered (SP)') 
    plt.ylabel('Accuracy')    
    # ax.set_ylim(0,1.00)
    fig.savefig((folder+title_type+'_SP_Accuracy.png'))
    plt.close(fig=fig)
    
def get_K_plot(accuracies,num_K,title_type=None):
    
    #Generate figure
    fig, ax = plt.subplots()
    ax.plot(np.array(num_K),np.array(accuracies))
    
    #Plot Accuracy for each value of K and annotate max value
    annot_max(np.array(num_K),np.array(accuracies),ax=ax)

    if title_type is not None:
        plt.title('{} Accuracy for {} values of K'.format(title_type,len(num_K)))
    else:
        plt.title('Accuracy for {} values of K'.format(len(num_K)))
   
    plt.xlabel('Number of Root Pixels Considered (K)') 
    plt.ylabel('Accuracy')    
    # ax.set_ylim(0,1.00)
    plt.show()
    
def get_cm(y_true, y_pred,plot_cm=False):
    confmatrix = confusion_matrix(y_true, y_pred, sample_weight=None)
    classes = np.unique(y_true)
    if plot_cm:
        plot_confusion_matrix(confmatrix, classes)
    return confmatrix
    
def plot_confusion_matrix_blue(cm, classes,
                         normalize=False,
                         title='Water',
                         cmap=plt.cm.Blues,
                         show_percent=True,ax=None,y_label='True label',
                         x_label='Predicted Label'):
   """
   This function prints and plots the confusion matrix.
   Normalization can be applied by setting normalize=True.
   """
   if normalize:
       cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  
   #Generate new figure if needed
   if ax is None:
       fig, ax = plt.subplots(figsize=(7,7))
   if show_percent:
       cm_percent = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
       im = ax.imshow(cm_percent,interpolation='nearest',cmap=cmap,vmin=0,vmax=100)
       plt.title(title)
       cb = plt.colorbar(im)
   else:
       im = ax.imshow(cm,interpolation='nearest',cmap=cmap)
       plt.title(title)
       cb = plt.colorbar(im)
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)
   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       if show_percent:
           s = str(format(cm[i, j], fmt) + '\n' + 
                   '(' + format(cm_percent[i,j],'.2f')+'%)')
       else:
           s = format(cm[i, j], fmt)
       ax.text(j, i, s,
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
   ax.set(xticks=np.arange(len(classes)),
                   yticks=np.arange(len(classes)),
                   xticklabels=classes,
                   yticklabels=classes,
                   ylabel=y_label,
                   xlabel=x_label)
   ax.set_ylim((len(classes) - 0.5, -0.5))
   plt.setp(ax.get_xticklabels(), rotation=45)
   plt.ylabel(y_label)
   plt.xlabel(x_label)
   plt.grid(False)
   plt.tight_layout()
   
def plot_contingency_table(cm, classes,cluster_labels,
                         normalize=False,
                         title='Water',
                         cmap=plt.cm.Blues,
                         show_percent=True,ax=None):
   """
   This function prints and plots the confusion matrix.
   Normalization can be applied by setting normalize=True.
   """
   if normalize:
       cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

   #Generate new figure if needed
   if ax is None:
       fig, ax = plt.subplots(figsize=(7,7))
   if show_percent:
       cm_percent = 100*cm.astype('float') / cm.sum(axis=0)[np.newaxis,:]
       im = ax.imshow(cm_percent,interpolation='nearest',cmap=cmap,vmin=0,vmax=100)
       plt.title(title)
       im_ratio = cm.shape[0]/cm.shape[1]
       cb = plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04)
   else:
       im = ax.imshow(cm,interpolation='nearest',cmap=cmap)
       plt.title(title)
       cb = plt.colorbar(im,fraction=0.046, pad=0.04)
   x_tick_marks = np.arange(len(classes))
   y_tick_marks = np.arange(len(cluster_labels))
   plt.xticks(x_tick_marks, classes, rotation=45)
   plt.yticks(y_tick_marks, cluster_labels)
   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       if show_percent:
           s = str(format(cm[i, j], fmt) + '\n' + 
                   '(' + format(cm_percent[i,j],'.2f')+'%)')
       else:
           s = format(cm[i, j], fmt)
       ax.text(j, i, s,
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
   ax.set(xticks=np.arange(len(classes)),
                   yticks=np.arange(len(cluster_labels)),
                   xticklabels=classes,
                   yticklabels=cluster_labels,
                   ylabel="Cluster Labels",
                   xlabel="True Treatment")
   ax.set_ylim((len(cluster_labels) - 0.5, -0.5))
   ax.set_xlim((len(classes) - 0.5, -0.5))
   plt.setp(ax.get_xticklabels(), rotation=45)
   plt.ylabel("Cluster Labels")
   plt.xlabel("True Treatment")
   plt.grid(False)
   plt.tight_layout()
   
def plot_norm_table(cm, features,normalization,
                      normalize=False,
                      title='Water',
                      cmap=plt.cm.Blues,
                      show_percent=False,ax=None,SP=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
 
    #Generate new figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,7))
    if show_percent:
        cm_percent = 100*cm.astype('float') / cm.sum(axis=0)[np.newaxis,:]
        im = ax.imshow(cm_percent,interpolation='nearest',cmap=cmap,vmin=0,vmax=100)
        plt.title(title)
        im_ratio = cm.shape[0]/cm.shape[1]
        cb = plt.colorbar(im,fraction=0.046*im_ratio, pad=0.04)
    else:
        im = ax.imshow(cm,interpolation='nearest',cmap=cmap)
        plt.title(title)
        cb = plt.colorbar(im,fraction=0.046, pad=0.04)
    x_tick_marks = np.arange(len(features))
    y_tick_marks = np.arange(len(normalization))
    plt.xticks(x_tick_marks, features, rotation=45)
    plt.yticks(y_tick_marks, normalization)
    fmt = '.0f' if SP else'.4f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if show_percent:
            s = str(format(cm[i, j], fmt) + '\n' + 
                    '(' + format(cm_percent[i,j],'.2f')+'%)')
        else:
            s = format(cm[i, j], fmt)
        ax.text(j, i, s,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ax.set(xticks=np.arange(len(features)),
                    yticks=np.arange(len(normalization)),
                    xticklabels=features,
                    yticklabels=normalization,
                    ylabel="Features",
                    xlabel="Downsampling")
    ax.set_ylim((len(normalization) - 0.5, -0.5))
    ax.set_xlim((len(features) - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.ylabel("Features")
    plt.xlabel("Downsampling")
    plt.grid(False)
    plt.tight_layout()
   

def plot_avg_norm_table(cm, features,normalization,
                      normalize=False,
                      title='Water',
                      cmap=plt.cm.Blues,
                      show_percent=False,ax=None,SP=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #Compute average CM values//
    std_cm = np.int64(np.ceil(np.std(cm,axis=2)))
    cm = np.int64(np.ceil(np.mean(cm,axis=2)))
    #Generate new figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    if show_percent:
        cm_percent = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_percent_std = 100*std_cm.astype('float') / std_cm.sum(axis=1)[:, np.newaxis]
        im = ax.imshow(cm_percent,interpolation='nearest',cmap=cmap,vmin=0,vmax=100)
        plt.title(title)
        cb = plt.colorbar(im)
    else:
        im = ax.imshow(cm,interpolation='nearest',cmap=cmap)
        plt.title(title)
        cb = plt.colorbar(im)
        # cb.set_clim(0,100)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    x_tick_marks = np.arange(len(features))
    y_tick_marks = np.arange(len(normalization))
    plt.xticks(x_tick_marks, features, rotation=45)
    plt.yticks(y_tick_marks, normalization)
    fmt = '.0f' if SP else'.4f'
    thresh = cm.max() / 2.
 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if show_percent:
            s = str(format(cm[i, j], fmt) + '±' + format(std_cm[i, j], fmt) +
                    '\n' + '(' + format(cm_percent[i,j],'.2f') + '±' + 
                    format(cm_percent_std[i,j],'.2f') + '%)')
        else:
            s = str(format(cm[i, j], fmt) + '±' + format(std_cm[i, j], fmt))
        ax.text(j, i, s,
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    ax.set(xticks=np.arange(len(features)),
                    yticks=np.arange(len(normalization)),
                    xticklabels=features,
                    yticklabels=normalization,
                    ylabel="Features",
                    xlabel="Downsampling")
    ax.set_ylim((len(normalization) - 0.5, -0.5))
    ax.set_xlim((len(features) - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.ylabel("Features")
    plt.xlabel("Downsampling")
    plt.grid(False)
    plt.tight_layout()


#plot the training images that overlaid each other from two different classes
#individually and show the difference between them.
def plot_trainset(trained_model):
    #import pdb; pdb.set_trace()
    #plot overlaid image for each water level
    water_images = trained_model['X_train_water']
    water_levels = trained_model['water_levels']
    if water_images.shape[-1] == 1: 
        water_images = water_images.squeeze(-1)
        cmap = 'gray'
        
    else: 
        cmap = None
    
    fig_water = plt.figure()
    num_class_water = water_images.shape[0]
    for i in range(num_class_water):
        nor_water_images = (water_images[i] / water_images[i].max(0).max(0) *255).astype('int')
        fig_water.add_subplot(1, num_class_water, i+1)
        plt.imshow(nor_water_images, cmap=cmap)
        plt.title('Water=%.f%%' %(water_levels[i]*100), fontsize = 12)
        plt.colorbar()
        plt.axis('off')
    plt.show(block=True)
    fig_water.savefig('New_waterlevel.png', dpi=1000)
    
    
    #plot overlaid image for each cultivar
    cultivar_images = trained_model['X_train_cultivar']
    cultivar = trained_model['cultivar']
    if cultivar_images.shape[-1] == 1: 
        cultivar_images = cultivar_images.squeeze(-1)
        cmap = 'gray'
    else: 
        cmap = None
        
    fig_cultivar = plt.figure()
    num_class_cultivar = cultivar_images.shape[0]
    for i in range(num_class_cultivar):

        nor_cultivar_images = (cultivar_images[i] / cultivar_images[i].max(0).max(0) *255).astype('int')
        fig_cultivar.add_subplot(1, num_class_cultivar,i+1)
        plt.imshow(nor_cultivar_images, cmap=cmap)
        plt.axis('off')
        plt.title('Cultivar=%s' %(cultivar[i]), fontsize = 12)
        plt.colorbar()
        
    plt.show(block=True)
    fig_cultivar.savefig('New_cultivar.png', dpi=1000)

def plot_ori_clusters(dataset, water_level=None, cultivar=None):
    """
    This function plots all the images with specified labels (original labels).
    If no label specified, the images are grouped based on ground truth labels.
    """
    
    
    water_level_labels = np.unique(dataset['train']['water_level_labels'])
    cultivar_labels = np.unique(dataset['train']['cultivar_labels'])
    data = dataset['train']['data']
    
    # show images without specified labels
    if water_level is None and cultivar is None:
        ncols = 11 # how many images in each row
        # Plot original images according to water level labels
        for lab in water_level_labels:
            idx = np.where(dataset['train']['water_level_labels']==lab)[0]
            nrows = int(np.ceil(len(idx)/ncols))
            
            fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(20,8))
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.3, hspace=0.0)
            fig.suptitle('Image Clusters of Water Level %.f%%' %(lab*100))
            
            for i, axi in enumerate(ax.flat):
                # pdb.set_trace()
                if i >= len(idx): 
                    axi.set_axis_off()
                    continue
                img = data[idx[i]]
                img = 255 - img
                name = dataset['train']['train_names'][idx[i]]
                axi.imshow(img, cmap='gray')
                
                axi.set_title(name, fontsize=9)
                axi.set_xticks([])
                axi.set_yticks([])
            plt.show()
            
        # Plot original images according to cultivar label
        for lab in cultivar_labels:
            idx = np.where(dataset['train']['cultivar_labels']==lab)[0]
            nrows = int(np.ceil(len(idx)/ncols))
            
            fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(20,8))
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.3, hspace=0.0)
            fig.suptitle('Image Clusters of Cultivar %s' %(lab))
            
            for i, axi in enumerate(ax.flat):
                # pdb.set_trace()
                if i >= len(idx): 
                    axi.set_axis_off()
                    continue
                img = data[idx[i]]
                img = 255 - img
                name = dataset['train']['train_names'][idx[i]]
                axi.imshow(img, cmap='gray')
                
                axi.set_title(name, fontsize=10)
                axi.set_xticks([])
                axi.set_yticks([])
            plt.show()
    
    # show images with specified combination of labels
    else:
        ncols = 5
        wl_idx = np.where(dataset['train']['water_level_labels'] == water_level)[0]
        c_idx = np.where(dataset['train']['cultivar_labels'] == cultivar)[0]
        idx = np.intersect1d(wl_idx, c_idx)
        if len(idx) == 0:
            print('No such cross treatment.')
            return
        nrows = int(np.ceil(len(idx)/ncols))
        
        fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(20,8))
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.3, hspace=0.0)
        fig.suptitle('Image Clusters of Cultivar %s and Water Level of %.f%%' %(cultivar, water_level*100))
        
        for i, axi in enumerate(ax.flat):
            # pdb.set_trace()
            if i >= len(idx): 
                axi.set_axis_off()
                continue
            img = data[idx[i]]
            img = 255 - img
            name = dataset['train']['train_names'][idx[i]]
            axi.imshow(img, cmap='gray')
            
            axi.set_title(name, fontsize=10)
            axi.set_xticks([])
            axi.set_yticks([])
        plt.show()

def plot_true_label(X, images, labels, saveout, fig_dist=4e-2, 
                    embed_images=True, title=None, vis_fig_type='Image'):
    temp_labels = np.copy(np.array(labels))
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    class_names = np.unique(temp_labels).tolist()
    colors = colormap.rainbow(np.linspace(0, 1, len(class_names)))
    count = 0
    
    plt.figure(figsize=(20,10))
    ax1 = plt.subplot(111)
    for treatment in class_names:
        plt.scatter(X[np.where(temp_labels==treatment),0], X[np.where(temp_labels==treatment),1], 
                    color = colors[count,:], label=class_names[count])
        count += 1
    
    plt.legend(class_names, bbox_to_anchor=(1.04, 1), loc='upper left')
    
    if embed_images:
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < fig_dist:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                if vis_fig_type == 'Image':
                    imagebox = offsetbox.AnnotationBbox(
                        offsetbox.OffsetImage(resize(images[i]['Img'], (256,256)), 
                                              cmap=plt.cm.gray_r, zoom=0.2), X[i])
                if vis_fig_type == 'Feature':
                    imagebox = offsetbox.AnnotationBbox(
                        offsetbox.OffsetImage(resize(images[i]['Root_mask'], (256,256)), 
                                                     zoom=0.2), X[i])
                ax1.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    if saveout is not None:
        plt.savefig((saveout), dpi=300)
    plt.close()
    
def plot_global_fig(X, images, labels, saveout, fig_dist=4e-2, 
                    embed_images=True, title=None, vis_fig_type='Image'):
    
    temp_labels = np.copy(np.array(labels))
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    class_names = np.unique(temp_labels).tolist()
    colors = colormap.rainbow(np.linspace(0, 1, len(class_names)))
    count = 0
    
    plt.figure(figsize=(20,10))
    ax1 = plt.subplot(111)
    for treatment in class_names:
        plt.scatter(X[np.where(temp_labels==treatment),0], X[np.where(temp_labels==treatment),1], 
                    color = colors[count,:], label=class_names[count])
        count += 1
    
    plt.legend(class_names, bbox_to_anchor=(1.04, 1), loc='upper left')
    
    if embed_images:
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < fig_dist:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                if vis_fig_type == 'Image':
                    imagebox = offsetbox.AnnotationBbox(
                        offsetbox.OffsetImage(resize(images[i], (256,256)), 
                                              cmap=plt.cm.gray_r, zoom=0.2), X[i])
   
                ax1.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    if saveout is not None:
        plt.savefig((saveout), dpi=300)
    plt.close()
    
def plot_global_feats(dataset,feature,folder,seed=None,root_only=True):
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    
    #For each image in the dataset, compute the global feture value
    temp_imgs = np.concatenate((dataset['train']['data'],dataset['test']['data']))
    temp_water_levels = np.concatenate((dataset['train']['water_level_labels'], 
                                        dataset['test']['water_level_labels']))
    temp_cultivar_levels = np.concatenate((dataset['train']['cultivar_labels'],
                                           dataset['test']['cultivar_labels']))
    feat_vects = np.zeros((len(temp_imgs),2))
    
    count = 0
    for img in temp_imgs:
        
        if feature == 'fractal':
            feat = fractal_dimension(img,root_only=root_only)
        elif feature == 'lacunarity':
            feat = fractal_dimension(img,root_only=root_only,compute_lac=True)[-1]
        elif feature == 'root_pixels':
            #Computing number of root pixels
            feat = np.count_nonzero(img)/(img.size)
        
        feat_vects[count,0] = feat
        count += 1
    
    #Generate random y-axis values to space out points
    rng = np.random.default_rng(seed=seed)
    feat_vects[:,1] = rng.random(len(temp_imgs))
    labels = Get_Cluster_Labels(feat_vects[:,0].reshape(-1,1),seed=seed,split_data=False)
    # pdb.set_trace()
    
    #Generate plots for points
    #Cultivar
    plot_global_fig(feat_vects,temp_imgs,temp_cultivar_levels,folder+'Global_Cultivar_Labels.png',
                embed_images=False, title= 'Global {} Cultivar'.format(feature), vis_fig_type='Image')
    plot_global_fig(feat_vects,temp_imgs,temp_cultivar_levels,folder+'Global_Cultivar_Images.png',
                embed_images=True, title= 'Global {} Cultivar'.format(feature), vis_fig_type='Image')
    
    #Water levels
    plot_global_fig(feat_vects,temp_imgs,temp_water_levels,folder+'Global_Water_Levels_Labels.png',
                embed_images=False, title= 'Global {} Water Levels'.format(feature), vis_fig_type='Image')
    plot_global_fig(feat_vects,temp_imgs,temp_water_levels,folder+'Global_Water_Levels_Images.png',
                embed_images=True, title= 'Global {} Water Levels'.format(feature), vis_fig_type='Image')
    
    #Cluster labels
    #Water levels
    plot_global_fig(feat_vects,temp_imgs,labels,folder+'Global_Cluster_Labels.png',
                embed_images=False, title= 'Global {} Cluster Levels'.format(feature), vis_fig_type='Image')
    plot_global_fig(feat_vects,temp_imgs,labels,folder+'Global_Cluster_Images.png',
                embed_images=True, title= 'Global {} Cluster Labels'.format(feature), vis_fig_type='Image')
    # pdb.set_trace()
    
def Get_Cluster_Labels(EMD_mat,seed=42,split_data=True,
                           train_idx=None,test_idx=None,preferences=None):
    if split_data:
        EMD_train = EMD_mat[np.ix_(train_idx,train_idx)]
        preferences = preferences[train_idx]
    else:
        EMD_train = EMD_mat
    #Affinity propogation: 
    #preference set based on median of similarity of matrix
    # preferences = np.median(EMD_train_sim)*preferences
    af = AffinityPropagation(affinity='euclidean',random_state=seed,
                             preference=preferences).fit(EMD_train)
    cluster_centers_indices = af.cluster_centers_indices_
    labels_af = af.labels_

    #Get labels for testing data
    if split_data:
        test_labels = []
        for img in test_idx:
            #Grab EMD values from matrix
            temp_dists = EMD_mat[img,cluster_centers_indices]
            
            #Select minimal distance and assign label
            #Error occurs when convergence does not happen
            try:
                temp_center = cluster_centers_indices[np.argmin(temp_dists)]
                test_labels.append(np.where(cluster_centers_indices==temp_center)[0][0])
            except:
                test_labels.append(labels_af[0])
        
        #Combine test labels with training labels
        test_labels = np.array(test_labels)
        labels_af = np.concatenate((labels_af,test_labels),axis=0)
        
    return labels_af
    
    
    
    