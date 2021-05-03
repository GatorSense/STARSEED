# Computer Engineering for in-Depth Characterization of Early Root Architecture: example on Sesamum indicum using Earth Mover’s Distance 


In this repository, we provide the paper (TBD) and code for our analysis approach using Earth Mover's Distance (EMD) for root architeture analysis.

## Installation Prerequisites

This code uses python (v3.7.6) and OpenCV. 
Please use [`Anaconda's OpenCV`](https://anaconda.org/conda-forge/opencv] to download necessary packages.

## Main Script to run

Run `Batch_Cluster_All_Image.py` in Python IDE (e.g., Spyder) or command line.
Results are saved out in figures and excel spreadsheets

## Main Functions

The `Batch_Cluster_All_Image.py` runs using the following functions. 

1. Generate superpixel representations of root images  

```scores, cluster_scores, true_labels = SP_Clustering(**Parameters)```

2. Peform relational clustering on images

 ```scores, cluster_scores, true_labels = Generate_Relational_Clustering(**Parameters)```

3. Generate stacked visual images based on labels 

```Stack_EMD_Visual(**Parameters)```


## Parameters
The parameters can be set in the following script:

```Parameters.py```

## Inventory

```
https://github.com/jpeeps67/Root_analysis

└── root dir
    ├── Data   //Folder that contains root images, please be sure data is in correct format.
        ├── Crop_Run4  // Root images at various days of planting (DAP), no fertilizer.
        ├── Crop_Run5  // Root images at various days of planting (DAP), no fertilizer.
        ├── Crop_Run6  // Root images at various days of planting (DAP), with fertilizer.
        ├── Crop_Run7  // Root images at various days of planting (DAP), with fertilizer.
        ├── TubeNumberKey.csv  // Contains information about root data (repetition, tube number, etc.) 
    ├── Batch_Cluster_All_Images.py   //Run this. Main file to cluster all root architectures.
    ├── Parameters.py // Parameters file for main script.
    ├── Batch_Cluster_Root_structures_HPG.py  // Code to perform cross validation for clustering (need to update).
    ├── Interactive_Visual_MU.py // Interactive visual to show embedding of root images (need to update).
    ├── Papers  // Links to related publications (will post our paper here).
    ├── Old_main_scripts  //old main scripts for previous approaches (currently not in use)
    └── Utils  //utility functions
        ├── Compute_EMD.py  // Compute Earth Mover's  (EMD) between two image signatures.
        ├── Compute_fractal_dim.py  // Compute fractal and lacunarity features.
        ├── Load_data.py  // Load root images from Data directory.
        ├── Relational_Clustering.py  // Perform relational clustering on images based on EMD.
        ├── Superpixel_Hist_Clustering.py  // Generate superpixel signatures for relational clustering. 
        ├── Visualization.py  // Functions to create visuals for results. 
        ├── Visualize_SP_EMD.py  // Visualize EMD scores and flows between images.
        ├── Visualize_Stacked_EMD.py  // Visualize EMD scores and flows between class/cluster representatives.
     
```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

This product is Copyright (c) 2021 J. Peeples, W. Xu, and A. Zare. All rights reserved.

## <a name="CitingHist"></a>Citing EMD Analysis Approach

If you use the analysis code, please cite the following reference using the following entry.

**Plain Text (TBD)**


**BibTex (TBD)**
```

```
