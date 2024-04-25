# Towards Fairness in Federated Data Clustering: Bridging the Gap between Diverse Data Distributions

This repository contains code for a project focused on fairness in Federated Data Clustering by using the concept of personalization.

## Project Structure
project-root/
│
├── main_algo/ # Main code implementation
│ ├── p-FClus.py
│
├── algo_baseline/ # Baseline algorithms
│ ├── k-FED.py # Implementation of baseline algorithm 1 (k-FED)
│ ├── MFC.py # Implementation of baseline algorithm 2 (MFC)
│
├── synthetic_dataset/ # Code to generate Synthetic Dataset
│ ├── data_gen_lil.py # When little overlap among gaussians through which we distribute data among various clients
│ ├── data_gen_lot.py # When lot overlap among gaussians through which we distribute data among various clients
| ├── data_gen_no.py # When no overlap among gaussians through which we distribute data among various clients
|
└── README.md # Project overview,usage instructions and datasets used


## Main Code (`main_algo` Folder)

The `main_algo` folder contains the primary codebase for our project, including the main algorithm implementation and utility functions.

- `p-FClus.py`: This script serves as the entry point for running our main algorithm.

## Baseline Algorithms (`algo_baseline` Folder)

The `algo_baseline` folder includes implementations of baseline algorithms used for comparison and benchmarking.

- `MFC.py`: Implementation of MFC algorithm proposed in [MFC: A Multishot Approach to Federated Data Clustering](https://ebooks.iospress.nl/doi/10.3233/FAIA230451)
- `k-FED.py`: Implementation of k-FED algorithm proposed in [Heterogeneity for the Win: One-Shot Federated Clustering](https://proceedings.mlr.press/v139/dennis21a.html).

## Datasets

We have utilized several datasets in this project. Below are the links to access each dataset:

- [Adult]([link_to_dataset1](https://docs.google.com/spreadsheets/d/112OQQZbZ9ApnFiW986vSropX8Xa_akc5T1umAFLCawU/edit?usp=sharing))
- [Bank]([link_to_dataset2](https://docs.google.com/spreadsheets/d/1qltcW9vjPd1AgobqVheQtymzrzn7TVQPYWWtvAuEcdc/edit?usp=sharing))
- [CELEB-A]([link_to_dataset2](https://github.com/TalwalkarLab/leaf/tree/master/data/celeba))
- [Diabetes]([link_to_dataset2](https://docs.google.com/spreadsheets/d/1AZ433lHb3Dhq5EJu2a3c-GHQIomSiq0yTCbw3nVphhs/edit?usp=sharing))
- [FEMNIST]([link_to_dataset2](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist))
- [FMNIST]([link_to_dataset2](https://github.com/zalandoresearch/fashion-mnist/tree/master))
- [WISDM]([link_to_dataset2](https://www.cis.fordham.edu/wisdm/dataset.php))
## Usage

To run the main algorithm, navigate to the `main_algo` directory and execute `p-FClus.py`. Ensure that all dependencies are installed before running the script.

```bash
cd main_algo/
python main_script.py
