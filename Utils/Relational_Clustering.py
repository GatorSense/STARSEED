# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 20:21:52 2020
Function to generate embeddings using TSNE and/or UMAP; use embeddings to 
visualize cluster assignments of data
@author: jpeeples
"""

from sklearn.cluster import AffinityPropagation
from sklearn.manifold import TSNE, MDS, Isomap, locally_linear_embedding
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics, preprocessing
from sklearn.utils import check_X_y
from matplotlib import offsetbox
import umap
import matplotlib.pyplot as plt
import numpy as np
import itertools
from itertools import cycle
from skimage.transform import resize
from Utils.Visualization import plot_contingency_table, plot_confusion_matrix_blue, plot_true_label
from Utils.Visualize_SP_EMD import Visualize_EMD
from Utils.Visualize_Stacked_EMD import Stack_EMD_Visual
from Utils.Compute_EMD import compute_EMD
import os
import pdb

def compute_scatter_metric(EMD_mat, labels):
    uniq_label = np.unique(labels)
    
    score = 0
    for item in uniq_label:
        # pdb.set_trace()
        intra_idx = np.where(labels == item)
        inter_idx = np.where(labels != item)
        intra_dis = EMD_mat[intra_idx[0],:][:,intra_idx[0]]
        inter_dis = EMD_mat[intra_idx[0],:][:,inter_idx[0]]
        # score += intra_dis.mean()/inter_dis.mean() #Minimize
        score += inter_dis.mean()/intra_dis.mean() #Maximize
    
    return score

def clusterwise_calinski_harabasz_score(X, labels):
    """Compute the Calinski and Harabasz score for each cluster.
    It is also known as the Variance Ratio Criterion.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.
    Returns
    -------
    score : float
        The resulting Calinski-Harabasz score.
    References
    ----------
    .. [1] `T. Calinski and J. Harabasz, 1974. "A dendrite method for cluster
       analysis". Communications in Statistics
       <https://www.tandfonline.com/doi/abs/10.1080/03610927408827101>`_
    """
    X, labels = check_X_y(X, labels)
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)

    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    metrics.cluster._unsupervised.check_number_of_labels(n_labels, n_samples)

    score = np.zeros((n_labels+1))
    mean = np.mean(X, axis=0)
    for k in range(n_labels):
        extra_disp, intra_disp = 0., 0.
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp = len(cluster_k) * np.sum((mean_k - mean) ** 2)
        intra_disp = np.sum((cluster_k - mean_k) ** 2)
        if intra_disp == 0.:
            score[k] = 1. 
        else:
            score[k] = extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.))
    
    score[-1] = metrics.calinski_harabasz_score(X, labels)
    return score

def compute_metric(EMD_mat, labels, func='Scatter', seed=None, embedding=None,
                   clusterwise=True):
    """
    Parameters
    ----------
    EMD_mat : np.arrary. EMD matrix
    labels : np.array. The label of each sample
    func : str. 'scatter' is custom'compute_scatter_metric' function
                'silhouette' is silhouette score from sklean.
                'Calinski-Harabasz' is Calinski-Harabasz score from sklearn
    seed: int. random seed for silhouette score
    embedding: MDS (or other) projection of EMD dissimilarity matrix

    """
    # import pdb; pdb.set_trace()
    if func == 'Scatter':
        score = compute_scatter_metric(EMD_mat, labels)
    elif func == 'Silhouette':
        score = metrics.silhouette_score(EMD_mat, labels, metric='precomputed',
                                                  random_state=seed)
    elif func == 'Calinski-Harabasz':
        if clusterwise:
            score = clusterwise_calinski_harabasz_score(embedding, labels)
        elif not clusterwise:
            score = metrics.calinski_harabasz_score(embedding, labels)
    
    else:
        raise RuntimeError('Invalid metric, please select Scatter, Silhoutte, or Calinski-Harabasz')
    return score


def plot_components(data, proj, images=None, ax=None,
                    thumb_frac=0.2, cmap='gray',indices=None,treatments=None,
                    cluster_center=None,cluster_index=0,cluster_treatment=None,
                    vis_fig_type='Image'):
    ax = ax or plt.gca()
    
    if vis_fig_type == 'Image':
        cmap = 'binary'
    if vis_fig_type == 'Feature':
        cmap = None
    if indices is None:
            indices = np.arange(data.shape[0])
            
    ax.plot(proj[indices, 0], proj[indices, 1], '.k')
    count=0
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in indices:
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2 and cluster_center is None: #Add constraint to always plot center
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            # #Rescale images to be 0 to 255
            if treatments is not None:
                ax.text(proj[i][0],proj[i][1], 
                        'Cluster '+str(count)+': ' + treatments[count], 
                        bbox=dict(fill=False, edgecolor='red', linewidth=2))
                # annotate = ax.annotate('Cluster Center '+str(count)+':' + treatments[count],  
                #             xy=(proj[i][0], proj[i][1]),color='white',fontsize="large",
                #             weight='heavy',horizontalalignment='center',
                #             verticalalignment='center')
                count += 1
            #Img sizes varied from (112,112) to (56,56) to (64,64)
            if vis_fig_type == 'Image':
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(resize(images[i]['Img'],(64,64)),
                                          zoom=.9, cmap=cmap),proj[i])
            if vis_fig_type == 'Feature':
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(resize(images[i]['Root_mask'],(64,64)),
                                          zoom=.9, cmap=cmap),proj[i])
            ax.add_artist(imagebox)
         
            #Cluster center should be last index
            if cluster_center is not None and cluster_center==i: #Add label to cluster center image
                ax.text(proj[i][0],proj[i][1], 'Cluster '+str(cluster_index)+
                        ' ' + cluster_treatment, 
                        bbox=dict(fill=False, edgecolor='red', linewidth=2))
                

            
def Create_Cluster_Figures(EMD_mat,embedding,folder,numSP,EMD_test=None,
                           ax=None,images=None,embed='TSNE',labels=None,
                           class_names=None,seed=42,split_data=True,
                           train_idx=None,test_idx=None,lab=False,
                           img_names=None,num_imgs=5,root_only=True,
                           preferences=None): 
    #Create folder for cluster center images
    cluster_folder = folder + '_Cluster_Center_Images/'
    visual_folder = folder + '_EMD_Visualization/'
    
    #Create folder to save figures
    if not os.path.exists(cluster_folder):
        os.makedirs(cluster_folder)
    
    #Convert distance matrix to similarity matrix for clustering
    if split_data:
        EMD_train = EMD_mat[np.ix_(train_idx,train_idx)]
        preferences = preferences[train_idx]
    else:
        EMD_train = EMD_mat
        
    EMD_train_sim = 1 - EMD_train/(np.max(EMD_train)+6e-10)
    # EMD_train_sim = -EMD_train


    #View pairwise distance and similarity matrix for data used for clustering
    fig_dif, ax_dif = plt.subplots()
    im = ax_dif.imshow(EMD_train)
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.title('Pairwise Distances')
    fig_dif.savefig(folder[:-4]+'EMD_Distances')
    plt.close()
    
    fig_sim, ax_sim = plt.subplots()
    im = ax_sim.imshow(EMD_train_sim)
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.title('Pairwise Similarities')
    fig_sim.savefig(folder+'_EMD_Similarities',dpi=fig_sim.dpi)
    plt.close()
   
    #Affinity propogation: 
    #preference set based on median of similarity of matrix
    af = AffinityPropagation(affinity='precomputed',random_state=seed,
                             preference=preferences).fit(EMD_train_sim)
    cluster_centers_indices = af.cluster_centers_indices_
    labels_af = af.labels_
    n_clusters_ = len(cluster_centers_indices)

    #If no clusters are found, skip visualizations
    if not(n_clusters_== 0):
        #Get labels for testing data
        img_count = num_imgs
        if split_data:
            test_labels = []
            for img in test_idx:
                #Grab EMD values from matrix (in future, compute distance and flow matrix)
                #Can add visualization here
                temp_dists = EMD_mat[img,cluster_centers_indices]
                
                #Visualize test sample to each cluster center
                flow_matrices = []
                
                for train_center in cluster_centers_indices:
                    #Compute flow matrix
                    _, temp_flow =compute_EMD(images[img]['SP_profile'],
                                                     images[train_center]['SP_profile'],
                                                     root_only=root_only)
                    flow_matrices.append(temp_flow)
                   
                #Create visualizations
                #For loop is to have parenthesis for class names (remove later)
                temp_class_names = []
                temp_class_imgs = []

                for idx in cluster_centers_indices.astype(int):
                    temp_class_names.append(class_names[labels[idx].astype(int)])
                    temp_class_imgs.append(img_names[idx])
                    
                    
                labels = labels.astype(int)
                if img_count > 0:
                    #Width for root pixels/fractal - 80, lacunarity- 
                    Visualize_EMD(images[img],np.array(images)[cluster_centers_indices.astype(int)],
                                  temp_dists,flow_matrices,visual_folder,class_names[labels[img]],
                                  img_names[img], num_class=n_clusters_,
                                  title='EMD Cluster Centers for Test Image {}'.format(img_names[img]),
                                  class_names=temp_class_names,
                                  lab=lab,sp_overlay=True,train_imgs_names=temp_class_imgs,
                                  arrow_width_scale=80,cmap='binary')
                    
                    Visualize_EMD(images[img],np.array(images)[cluster_centers_indices.astype(int)],
                                  temp_dists,flow_matrices,visual_folder,class_names[labels[img]],
                                  img_names[img], num_class=n_clusters_,
                                  title='EMD Cluster Centers for Test Image {}'.format(img_names[img]),
                                  class_names=temp_class_names,
                                  lab=lab,sp_overlay=False,train_imgs_names=temp_class_imgs,
                                  arrow_width_scale=80,cmap='binary')
                    img_count -= 1
                    
                # pdb.set_trace()
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
            
        if ax is None:
            fig,ax = plt.subplots(1,1)
            plt.subplots_adjust(right=.75)
    
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        center_treatments = []
    
        for k, col in zip(range(n_clusters_), colors):
            class_members = labels_af == k
            cluster_center = embedding[cluster_centers_indices[k]]
            ax.plot(embedding[class_members, 0], embedding[class_members, 1], col + '.')
            if labels is None:
                ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                          markeredgecolor='k', markersize=14,label='Cluster '+str(k)+':')
            else:
                labels = labels.astype(int)
                treatment = str(class_names[labels[cluster_centers_indices[k]]])
                center_treatments.append(treatment)
                ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                          markeredgecolor='k', markersize=14,label=treatment)
                ax.annotate(str(k),  xy=(cluster_center[0], cluster_center[1]), color='white',
                fontsize="large", weight='heavy',
                horizontalalignment='center',
                verticalalignment='center')
            for x in embedding[class_members]:
                ax.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], color = col)
            #Plot testing points    
            if split_data:
               test_pts = np.concatenate((np.zeros(len(train_idx)),np.ones(len(test_idx))),axis=0).astype(bool)
               test_class_members = np.logical_and(class_members,test_pts)
               ax.plot(embedding[test_class_members,0], embedding[test_class_members,1], col + '*')
            
            #For each cluster center, plot root images in cluster
            if split_data:
                fig_cluster,ax_cluster = plt.subplots(1,2,figsize=(14,7))
                #Testing images
                plot_components(EMD_mat,embedding,images=images,ax=ax_cluster[1],cmap='binary',
                                thumb_frac=0.05,indices=np.arange(0,len(class_members))[test_class_members],
                                treatments=None,cluster_center=cluster_centers_indices[k],
                                cluster_index=k,cluster_treatment=treatment)
                
                #Training images
                train_pts = np.concatenate((np.ones(len(train_idx)),np.zeros(len(test_idx))),axis=0).astype(bool)
                train_class_members = np.logical_and(class_members,train_pts)
                plot_components(EMD_mat,embedding,images=images,ax=ax_cluster[0],cmap='binary',
                                thumb_frac=0.05,indices=np.arange(0,len(class_members))[train_class_members],
                                treatments=None,cluster_center=cluster_centers_indices[k],
                                cluster_index=k,cluster_treatment=treatment)
                ax_cluster[0].axis('off')
                ax_cluster[0].set_title('Training Images',y=1.08)
                ax_cluster[1].axis('off')
                ax_cluster[1].set_title('Testing Images',y=1.08)
                fig_cluster.suptitle(embed + ' Embedding of AP Cluster Center ' + str(k) +': ' +
                                     treatment + ' for ' + str(numSP) + ' Superpixels',y=.995)
                fig_cluster.savefig((cluster_folder+'Root_Images_AP_Cluster_{}.png'.format(k)))
                plt.close(fig=fig_cluster)
               
                
            else:
                fig_cluster,ax_cluster = plt.subplots(1,1,figsize=(14,7))
                plot_components(EMD_train,embedding,images=images,ax=ax_cluster,cmap='binary',
                                thumb_frac=0.05,indices=np.arange(0,len(class_members))[class_members],
                                treatments=None,cluster_center=cluster_centers_indices[k],
                                cluster_index=k,cluster_treatment=treatment)
                #Bug with this, may add labels to images
                #class_names[labels[np.arange(0,len(class_members))][class_members]]
                ax_cluster.axis('off')
                ax_cluster.set_title(embed + ' Embedding of AP Cluster Center ' + str(k) +': ' +
                                     treatment + ' for ' + str(numSP) 
                             + ' Superpixels',y=1.08)
                fig_cluster.savefig((cluster_folder+'Root_Images_AP_Cluster_{}.png'.format(k)))
                plt.close(fig=fig_cluster)
       
        ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
        plt.title(('Affnity Propogation: Est {:d} clusters with ' +
                   '{:d} Superpixels').format(n_clusters_,numSP),y=1.08)
        plt.savefig((folder+'_Aff_Prop.png'))
        plt.close()
        
        #Plot images that are cluster centers
        if images is not None:
            fig,ax = plt.subplots(1,1,figsize=(14,7))
            plot_components(EMD_train,embedding,images=images,ax=ax,cmap='binary',
                            thumb_frac=0.05,indices=cluster_centers_indices,
                            treatments=center_treatments)
            ax.axis('off')
            ax.set_title(embed + ' Embedding of AP Cluster Centers for ' + str(numSP) 
                         + ' Superpixels',y=1.08)
            fig.savefig((folder+'_Root_Images_AP_Centers.png'))
            plt.close(fig=fig)
    else:
        print('No Clusters Found')
        if split_data:
            test_labels = -np.ones(len(test_idx))
            labels_af = np.concatenate((labels_af,test_labels),axis=0)
        else:
            pass
    
    return {'Affinity_Propagation': labels_af}
    
def Create_Contigency_Table(class_labels,cluster_labels,folder,numSP,ct_title,
                            treatment,test_idx=None,adjusted=True,set_fig_size=True,
                            cm_title=' Cultivar and Water Levels'):
    
    #Get table
    if test_idx is not None:
        cluster_labels = np.asarray(cluster_labels)[test_idx]
        class_labels = np.asarray(class_labels)[test_idx]
    
    
    ct = metrics.cluster.contingency_matrix(cluster_labels,class_labels)
    pair_cm = metrics.cluster.pair_confusion_matrix(class_labels,cluster_labels)
    
    
    #Visualize
    if set_fig_size:
        fig, ax = plt.subplots(1,1,figsize=(12,6))
    else:
        fig, ax = plt.subplots(1,1)
        
    plot_contingency_table(ct,treatment,np.unique(cluster_labels).tolist(),
                           title=(ct_title+' {:d} Superpixels').format(numSP),ax=ax)
    
    fig_cm, ax_cm = plt.subplots(1,1)
    plot_confusion_matrix_blue(pair_cm,['No','Yes'],title=(cm_title + ' {:d} Superpixels').format(numSP),
                               ax=ax_cm, y_label='Same ' + cm_title, x_label='Same Cluster')
 
    # #Compute metrics
    if adjusted:
        # scores = []
        # scores.append(metrics.adjusted_rand_score(class_labels,cluster_labels))
        # scores.append(metrics.adjusted_mutual_info_score(class_labels,cluster_labels))
        # scores.append(metrics.fowlkes_mallows_score(class_labels,cluster_labels))
        
        #Compute precision, recall, and f1 scores based on pair cm
        precision = pair_cm[1,1]/ np.sum(pair_cm[:,1])
        recall = pair_cm[1,1]/ np.sum(pair_cm[1,:])
        f1 = 2 * (precision * recall) / (precision + recall)
        scores = [precision, recall, f1]
    else:
        scores = metrics.homogeneity_completeness_v_measure(class_labels,
                                                            cluster_labels,
                                                            beta=1)
        
    
    #May add adjusted rand index
    scores = list(scores)
    # scores.append(metrics.adjusted_rand_score(class_labels,cluster_labels))
    np.savetxt((folder + ct_title + '_Cluster_scores.txt'), scores, fmt='%.4f')
    
    #Save figure and close
    fig.savefig((folder+ct_title+'_Contigency_Table.png'),dpi=fig.dpi)
    fig_cm.savefig((folder+ct_title+'_Pair_Confusion.png'),dpi=fig_cm.dpi)
    plt.close('all')
    
    
    return scores

def Get_Cluster_Labels(EMD_mat,seed=42,split_data=True,
                           train_idx=None,test_idx=None,preferences=None):
    

    if split_data:
        EMD_train = EMD_mat[np.ix_(train_idx,train_idx)]
        preferences = preferences[train_idx]
    else:
        EMD_train = EMD_mat
        
    #Add small positive value incase all values 0
    EMD_train_sim = 1 - EMD_train/(np.max(EMD_train)+6e-10)
    
    #Affinity propogation: 
    #preference set based on median of similarity of matrix
    # preferences = np.median(EMD_train_sim)*preferences
    af = AffinityPropagation(affinity='precomputed',random_state=seed,
                             preference=preferences).fit(EMD_train_sim)
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
    
def Generate_Relational_Clustering(EMD_mat,images,cultivar_labels,
                                   water_level_labels,names,split_data=True,
                                   folder='Cluster_Imgs_SP/',embed='TSNE',numSP=100,
                                   seed=42,train_idx=None,test_idx=None,
                                   num_imgs=5,embed_only=False,root_only=True,
                                   preferences=None,adjusted=True,num_neighbors=15,
                                   label_type='Cluster', score_metric='Scatter',
                                   vis_fig_type='Image',features='fractal'):
    #Create folder to save figures
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    #Change preferences to array
    if preferences is not None:
        preferences = np.stack(preferences,axis=0)

    #Compute embeddings
    #For TSNE, compute embedding with training and testing data (only for 
    # visualization and not clustering approach)
    if embed == 'TSNE':
        embed_method = TSNE(n_components=2,verbose=0,random_state=seed,
                          metric='precomputed')
        embed_method = embed_method.fit(EMD_mat)
        embedding = embed_method.embedding_
        error = embed_method.kl_divergence_
        
        
    elif embed == 'UMAP': 
        if label_type == 'Cluster':
            #Cluster labels
            labels = Get_Cluster_Labels(EMD_mat,seed=seed,split_data=split_data,
                                        train_idx=train_idx,test_idx=test_idx,preferences=preferences)
        #Cultivar
        elif label_type == 'Cultivar':
            le = preprocessing.LabelEncoder()
            labels = le.fit_transform(cultivar_labels)
        #Water level
        elif label_type == 'Water_Levels':
            le = preprocessing.LabelEncoder()
            labels = le.fit_transform(water_level_labels)
        #Cross Treatments
        elif label_type == 'Cross_Treatments':
            class_names = list(itertools.product(list(np.unique(cultivar_labels)),
                                                  list(np.unique(water_level_labels))))
            count = 0
            labels = np.zeros(len(cultivar_labels))
            for treatment in class_names:
                temp_cultivar = np.where(np.array(cultivar_labels)==treatment[0])
                temp_water_level = np.where(np.array(water_level_labels)==treatment[1])
                intersection = np.intersect1d(temp_cultivar,temp_water_level)
                labels[intersection] = count
                count += 1
        
        #No labels
        else:
            labels = None
        
        embedding = umap.UMAP(n_neighbors=num_neighbors,min_dist=.5,
                              random_state=seed,metric='precomputed').fit_transform(EMD_mat,y=labels)
        
        #Change embed name to UMAP_label_type
        embed = embed + '_' + label_type
        error = 0 #How to compute objective value
    
    elif embed == 'MDS':
        embed_method = MDS(n_components=2,verbose=0,random_state=seed,
                        dissimilarity='precomputed', metric=True)
        embed_method = embed_method.fit(EMD_mat)
        embedding = embed_method.embedding_
        error = embed_method.stress_
        #Save score
        with open(folder + 'MDS_Score.txt', 'w') as f:
          f.write(str(np.round(error,decimals=4)))
  
    elif embed == 'ISOMAP':
        #Eigen_solver set to dense (repeatable)
        # pdb.set_trace()
        embed_method = Isomap(n_neighbors=num_neighbors,eigen_solver='dense',
                              metric='precomputed')
        embed_method = embed_method.fit(EMD_mat)
        embedding = embed_method.embedding_
        error = embed_method.reconstruction_error()
  
    elif embed == 'LLE':
        #Create nearest neighbors object (LLE cannot operate on distance matrix)
        neigh = NearestNeighbors(n_neighbors=num_neighbors,metric='precomputed')
        neigh.fit(EMD_mat)
        embedding, error = locally_linear_embedding(neigh,n_neighbors=num_neighbors,n_components=2,
                                              method='standard',
                                              random_state=seed,eigen_solver='dense')
    else:
        assert \
            f'Visualization option not currently supported'
        
    #If Embed only break
    if embed_only:
        return embedding
    
    #Plot true labels for each treatment
    xy = embedding[np.arange(len(cultivar_labels))]
    plot_true_label(xy, images, cultivar_labels, 
                    saveout=folder+embed+'_Cultivar_True_Labels.png', 
                    fig_dist=4e-2, title='Cultivar True Labels ' + str(numSP) + ' Superpixels',
                    vis_fig_type=vis_fig_type)
    
    xy = embedding[np.arange(len(water_level_labels))]
    plot_true_label(xy, images, water_level_labels, 
                    saveout=folder+embed+'_Water_Levels_True_Labels.png', 
                    fig_dist=4e-2, title='Water Levels True Labels ' + str(numSP) + ' Superpixels',
                    vis_fig_type=vis_fig_type)
    
    fig3, ax3 = plt.subplots(1,1)
    plt.subplots_adjust(right=.75)
    class_names = list(itertools.product(list(np.unique(cultivar_labels)),
                                          list(np.unique(water_level_labels))))
    markers =['o','*','s','^']
    plt_colors = ['red','green','blue','purple']
    color_markers = list(itertools.product(markers,plt_colors))
    count = 0
    cross_labels = np.zeros(len(cultivar_labels))
    for treatment in class_names:
        
        temp_cultivar = np.where(np.array(cultivar_labels)==treatment[0])
        temp_water_level = np.where(np.array(water_level_labels)==treatment[1])
        intersection = np.intersect1d(temp_cultivar,temp_water_level)
        cross_labels[intersection] = count
        x = embedding[[intersection],0]
        y = embedding[[intersection],1]
        ax3.scatter(x, y, color = color_markers[count][1], marker = color_markers[count][0]
                       , label=class_names[count])
        count += 1
    ax3.set_title('Cross Treatments True Labels for ' + str(numSP) + ' Superpixels')
    ax3.legend(class_names,bbox_to_anchor=(1.04, 1), borderaxespad=0.)
    fig3.savefig((folder+embed+'_Cross_Treatments_True_Labels.png'),dpi=fig3.dpi)
    plt.close(fig=fig3)
    
    fig,ax = plt.subplots(1,1,figsize=(14,7))
    plot_components(EMD_mat,embedding,images=images,ax=ax,
                    vis_fig_type=vis_fig_type,thumb_frac=0.05)
    ax.axis('off')
    ax.set_title(embed + ' Embedding for ' + str(numSP) + ' Superpixels',y=1.08)
    fig.savefig((folder+embed+'_Root_Images'))
    plt.close(fig=fig)
    
    #Generate cluster figures and get labels
    cluster_labels = Create_Cluster_Figures(EMD_mat,embedding,folder+'/'+embed,
                                            numSP,ax=None,images=images,embed=embed,
                                            labels = cross_labels,class_names=class_names,
                                            split_data=split_data,train_idx=train_idx,
                                            test_idx=test_idx,img_names=names,
                                            num_imgs=num_imgs,root_only=root_only,
                                            preferences=preferences)
    
    plt.close('all')
    
    #Compute Silhoutte score based on EMD matrix and true labels
    if split_data:
        #Compute on training data only
        EMD_train = EMD_mat[np.ix_(train_idx,train_idx)]
        water_levels_score = compute_metric(EMD_train, np.asarray(water_level_labels)[train_idx],
                                            func=score_metric, seed=seed, embedding=embedding[train_idx,:])
        cultivar_score = compute_metric(EMD_train, np.asarray(cultivar_labels)[train_idx],
                                        func=score_metric, seed=seed, embedding=embedding[train_idx,:])
        cross_score = compute_metric(EMD_train, np.asarray(cross_labels)[train_idx],
                                     func = score_metric, seed=seed, embedding=embedding[train_idx,:])
        try:
            cluster_score = compute_metric(EMD_train,
                                           cluster_labels['Affinity_Propagation'][train_idx],
                                           func=score_metric, seed=seed, embedding=embedding[train_idx,:])
        except:
            cluster_score = 0 #All points assigned to same cluster
        
    else:
        # import pdb; pdb.set_trace()
        water_levels_score = compute_metric(EMD_mat, water_level_labels,
                                            func=score_metric, seed=seed, embedding=embedding)
        le = preprocessing.LabelEncoder()
        cultivar_score = compute_metric(EMD_mat, le.fit_transform(cultivar_labels),
                                        func=score_metric, seed=seed, embedding=embedding)
        cross_score = compute_metric(EMD_mat, cross_labels,
                                     func=score_metric, seed=seed, embedding=embedding)
        try:
            cluster_score = compute_metric(EMD_mat, cluster_labels['Affinity_Propagation'],
                                           func=score_metric, seed=seed, embedding=embedding)
        except:
            cluster_score = 0 #All points assigned to same cluster
    
    EMD_scores = {'cultivar': cultivar_score, 'water_levels': water_levels_score,
              'cross_treatment': cross_score, 'clustering': cluster_score}
    
    #Visualize stacked results for each label type (only focused on cultivar and water levels)
    # labels = {'Cultivar': cultivar_labels, 'Water Level': water_level_labels, 
    #           'Cross Treatment': cross_labels, 'Cluster': cluster_labels['Affinity_Propagation']}
    labels = {'Cultivar': cultivar_labels, 'Water Level': water_level_labels}
    
    for key in labels:
        # if not(key == 'Cross Treatment'):
        #     class_names = None
        Stack_EMD_Visual(images,labels[key],folder+'Stacked_Image_Visuals/',
                         features,class_names=None,root_only=root_only,label_type=key)
    
    #Compute contigency tables and cluster metrics
    Cluster_scores = {}
    for key in cluster_labels:
        if split_data: #Compute for test only
            #Cultivar
            cultivar_cluster_scores = Create_Contigency_Table(cultivar_labels,
                                                              cluster_labels[key],
                                                              folder+'/',
                                    numSP,key+' Cultivar',
                                    np.unique(np.array(cultivar_labels)).tolist(),
                                    test_idx=test_idx,adjusted=adjusted,
                                    set_fig_size=False, cm_title= ' Cultivar')
            #Water
            water_level_cluster_scores = Create_Contigency_Table(water_level_labels,
                                                                 cluster_labels[key],
                                                                 folder+'/',
                                    numSP,key+' Water Levels',
                                    treatment=np.unique(np.array(water_level_labels)).tolist(),
                                    test_idx=test_idx,adjusted=adjusted,
                                    set_fig_size=False, cm_title= ' Water Levels')
            #Cross_treatment
            cross_treatments_cluster_scores = Create_Contigency_Table(cross_labels,
                                                                      cluster_labels[key],
                                                                      folder+'/',
                                    numSP,key+' Cross Treatments',
                                    treatment=class_names,test_idx=test_idx,
                                    adjusted=adjusted,cm_title=' Cultivar and Water Levels')
        else:
            #Cultivar
            le = preprocessing.LabelEncoder()
            cultivar_cluster_scores = Create_Contigency_Table(cultivar_labels,cluster_labels[key],folder+'/',
                                    numSP,key+' Cultivar',
                                    np.unique(np.array(cultivar_labels)).tolist(),
                                    adjusted=adjusted,
                                    set_fig_size=False,cm_title= ' Cultivar')
            #Water
            water_level_cluster_scores = Create_Contigency_Table(water_level_labels,cluster_labels[key],folder+'/',
                                    numSP,key+' Water Levels',
                                    treatment=np.unique(np.array(water_level_labels)).tolist(),
                                    adjusted=adjusted,
                                    set_fig_size=False, cm_title= ' Water Levels')
            #Cross_treatment
            cross_treatments_cluster_scores = Create_Contigency_Table(cross_labels,cluster_labels[key],folder+'/',
                                    numSP,key+' Cross Treatments',treatment=class_names,
                                    adjusted=adjusted, cm_title=' Cultivar and Water Levels')

        Cluster_scores[key] = {'Cultivar': cultivar_cluster_scores, 
                                'Water_Levels': water_level_cluster_scores,
                                'Cross_Treatment': cross_treatments_cluster_scores}
        
    return EMD_scores, Cluster_scores, labels
    
    
    
    
    
    
    
    