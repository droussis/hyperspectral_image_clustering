# Hyperspectral Image Unsupervised Classification

This project was conducted as part of the requirements of the "Clustering Algorithms" postgraduate course (Fall 2019) at the UoA (MSc in Data Science and Information Technologies).

The goal of this project is to compare different clustering algorithms for a land cover classification task. In particular, we identify the different crops that are depicted in an image (150x150) of the Salinas valley in California, USA. Each one of the pixels contains a spectral signature of 204 distinct spectral bands and is classified as one of the eight distinct types of crops in the valley.

___
## Additional Information
The repository contains a detailed report `project_report_roussis.pdf` which is necessary to understand the problem and all the steps that were taken. In particular, it contains information about the preprocessing steps that were used, the framework with which we compared the clustering algorithms, as well as general remarks on the performance of each one. The best configurations of the algorithms were tested in combination with **Principal Component Analysis** for dimensionality reduction. The following algorithms were compared:
- K-Means clustering
- Possibilistic C-Means clustering
- Fuzzy C-Means clustering
- Gaussian Mixture model (Probabilistic clustering)

Some of the clustering algorithms were implemented from scratch or were modified versions of functions provided by the instructors.

---
## Usage

  - MATLAB version: R2019b

Clone the repository and make sure to add both directories ('code' and 'data') to path in MATLAB. Then, simply execute any part of the `roussis_project.m` that you want to. Feel free to change any of the parameters or comment out the parts that are not of interest.

---
## License
The content of this repository is licensed under a MIT license. Please note that two functions (`pca_fun.m` and `k_means.m`) are provided under different licenses.
