# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 20:21:52 2020
Function to generate embeddings using TSNE and/or UMAP; use embeddings to 
visualize cluster assignments of data
@author: jpeeples
"""

from sklearn.manifold import TSNE, MDS, Isomap, locally_linear_embedding
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics, preprocessing
from sklearn.utils import check_X_y
from matplotlib import offsetbox
import umap
import matplotlib.pyplot as plt
import numpy as np
import itertools
from skimage.transform import resize
from Utils.Visualization import plot_true_label
from Utils.Visualize_Stacked_EMD import Stack_EMD_Visual
import os
import pdb


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
    if func == 'Silhouette':
        score = metrics.silhouette_score(EMD_mat, labels, metric='precomputed',
                                                  random_state=seed)
    elif func == 'Calinski-Harabasz':
        if clusterwise:
            score = clusterwise_calinski_harabasz_score(embedding, labels)
        elif not clusterwise:
            score = metrics.calinski_harabasz_score(embedding, labels)
    
    else:
        raise RuntimeError('Invalid metric, please select Silhoutte or Calinski-Harabasz')
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
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in indices:
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2 and cluster_center is None: #Add constraint to always plot center
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
        
            #Img sizes varied from (112,112) to (56,56) to (64,64)
            if vis_fig_type == 'Image':
                #TBD (add label color for roots)
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
                

    
def Generate_EMD_Clustering(EMD_mat,images,cultivar_labels,
                                   water_level_labels,names,split_data=True,
                                   folder='Cluster_Imgs_SP/',embed='TSNE',numSP=100,
                                   seed=42,train_idx=None,test_idx=None,
                                   embed_only=False,root_only=True,
                                   num_neighbors=15,label_type='Cluster', 
                                   score_metric='Scatter',vis_fig_type='Image',
                                   features='fractal'):
    #Create folder to save figures
    if not os.path.exists(folder):
        os.makedirs(folder)

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
        #Cultivar
        if label_type == 'Cultivar':
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
                    fig_dist=.15, title='Cultivar True Labels ' + str(numSP) + ' Superpixels',
                    vis_fig_type=vis_fig_type)
    
    xy = embedding[np.arange(len(water_level_labels))]
    plot_true_label(xy, images, water_level_labels, 
                    saveout=folder+embed+'_Water_Levels_True_Labels.png', 
                    fig_dist=.15, title='Water Levels True Labels ' + str(numSP) + ' Superpixels',
                    vis_fig_type=vis_fig_type,water = True)
    
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

    else:
        water_levels_score = compute_metric(EMD_mat, water_level_labels,
                                            func=score_metric, seed=seed, embedding=embedding)
        le = preprocessing.LabelEncoder()
        cultivar_score = compute_metric(EMD_mat, le.fit_transform(cultivar_labels),
                                        func=score_metric, seed=seed, embedding=embedding)
        cross_score = compute_metric(EMD_mat, cross_labels, func=score_metric, seed=seed, embedding=embedding)
    
    EMD_scores = {'cultivar': cultivar_score, 'water_levels': water_levels_score,'cross_treatment': cross_score}
    
    #Visualize stacked results for each label type (only focused on cultivar and water levels)
    labels = {'Cultivar': cultivar_labels, 'Water Level': water_level_labels}
    
    for key in labels:
        Stack_EMD_Visual(images,labels[key],folder+'Stacked_Image_Visuals/',
                         features,class_names=None,root_only=root_only,label_type=key)
    
 
        
    return EMD_scores, labels
    
    
    
    
    
    
    
    