# Spatial and Texture Analysis of Root System architEcture with Earth mover's Distance (STARSEED)


In this repository, we provide the [paper](https://plantmethods.biomedcentral.com/articles/10.1186/s13007-022-00974-z) and code for our analysis approach using Earth Mover's Distance (EMD) for root architeture analysis.

## Installation Prerequisites

This code uses python (v3.7.6) and OpenCV. 
Please use [`Anaconda's OpenCV`](https://anaconda.org/conda-forge/opencv) to download necessary packages.

## Main Script to run

Run `demo.py` in Python IDE (e.g., Spyder) or command line.
Results are saved out in figures and excel spreadsheets

## Main Functions

The `demo.py` runs using the following functions. 

1. Generate superpixel representations of root images  

```scores, cluster_scores, true_labels = SP_Clustering(**Parameters)```

2. Generate embeddings of images

 ```scores, cluster_scores, true_labels = Generate_Relational_Clustering(**Parameters)```

3. Generate stacked visual images based on labels 

```Stack_EMD_Visual(**Parameters)```


## Parameters
The parameters can be set in the following script:

```Parameters.py```

## Inventory

```
https://github.com/GatorSense/STARSEED

└── root dir
    ├── Data   //Folder that contains root images, please be sure data is in correct format.
        ├── Run1  // Root images at various days of planting (DAP), no fertilizer.
        ├── Run2  // Root images at various days of planting (DAP), no fertilizer.
        ├── Run3  // Root images at various days of planting (DAP), with fertilizer.
        ├── Run4  // Root images at various days of planting (DAP), with fertilizer.
        ├── TubeNumberKey.csv  // Contains information about root data (repetition, tube number, etc.) 
    ├── demo.py   //Run this. Main file to analyze root architectures.
    ├── Parameters.py // Parameters file for main script.
    ├── Record_time.py   // Generate supplemental figures computational time.
    └── Utils  //utility functions
        ├── Compute_EMD.py  // Compute Earth Mover's  (EMD) between two image signatures.
        ├── Compute_fractal_dim.py  // Compute fractal and lacunarity features.
        ├── Load_data.py  // Load root images from Data directory.
        ├── EMD_Clustering.py  // Generate embeddings of images based on EMD and return scores.
        ├── Superpixel_Hist_Clustering.py  // Generate superpixel signatures for relational clustering. 
        ├── Visualization.py  // Functions to create visuals for results. 
        ├── Visualize_SP_EMD.py  // Visualize EMD scores and flows between images.
        ├── Visualize_Stacked_EMD.py  // Visualize EMD scores and flows between class/cluster representatives.
     
```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

This product is Copyright (c) 2023 J. Peeples, W. Xu, R. Gloaguen, D. Rowland, A. Zare, and Z. Brym. All rights reserved.

## <a name="CitingHist"></a>Citing STARSEED

If you use the analysis code, please cite the following reference using the following entry.

**Plain Text**

J. Peeples, W. Xu, R. Gloaguen, D. Rowland, A. Zare, and Z. Brym, “Spatial and Texture Analysis of Root System Distri‑
bution with Earth Mover’s Distance (STARSEED),” in Plant Methods 19, 2023. doi: 10.1186/s13007‑022‑00974‑z.

**BibTex**
```
@article{peeples2023Spatial,
  title={Spatial and Texture Analysis of Root System Architecture with Earth Mover’s Distance(STARSEED)},
  author={Peeples, Joshua and Xu, Weihuang and Gloaguen, Romain and Rowland, Diane, and Zare, Alina and Brym, Zachary},
  journal={Plant Methods 19},
  year={2023},
  doi= {10.1186/s13007‑022‑00974‑z}
}
```

