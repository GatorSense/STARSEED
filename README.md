# Spatial and Texture Analysis of Root SystEm distribution with Earth mover’s Distance (STARSEED)


In this repository, we provide the [`paper`](https://www.biorxiv.org/content/10.1101/2021.08.31.458446) and code for our analysis approach using Earth Mover's Distance (EMD) for root sytem distribution analysis. GatorSense/STARSEED: Initial Release (Version v1.0). [`Zenodo`](https://doi.org/10.5281/zenodo.5364355) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5364355.svg)](https://doi.org/10.5281/zenodo.5364355)

## Installation Prerequisites

This code uses python (v3.7.6) and OpenCV. 
Please use [`Anaconda's OpenCV`](https://anaconda.org/conda-forge/opencv) to download necessary packages.

## Main Script to run

Run `demo.py` in Python IDE (e.g., Spyder) or command line.
Results are saved out in figures and excel spreadsheets.

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
    └── Utils  //utility functions
        ├── Compute_EMD.py  // Compute Earth Mover's (EMD) between two image signatures.
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

This product is Copyright (c) 2021 J. Peeples, W. Xu, R. Gloaguen, D. Rowland, A. Zare, and Z. Brym. All rights reserved.

## <a name="CitingHist"></a>Citing STARSEED

If you use the analysis code, please cite the following reference using the following entry.

**Plain Text**

Peeples, J., Xu, W., Gloaguen, R., Rowland, D., Zare, A., and Brym, Z. (2021). 
Spatial and Texture Analysis of Root SystEm distribution with Earth mover’s Distance (STARSEED).

**BibTex**
```
@article{peeples2021spatial,
  title={Spatial and Texture Analysis of Root SystEm distribution with Earth mover’s Distance},
  author={Peeples, Joshua and Xu, Weihuang and Gloaguen, Romain and Rowland, Diane and Zare, Alina and Brym, Zachary},
  journal={bioRxiv},
  year={2021},
  publisher={Cold Spring Harbor Laboratory}
}
```

