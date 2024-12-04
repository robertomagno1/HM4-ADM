# Algorithmic Methods for Data Mining - Homework 4 🎥

This is a GitHub repository created to submit the fourth homework of the **Algorithmic Methods for Data Mining (ADM)** course for the MSc. in Data Science at the Sapienza University of Rome.

---

### Files and Directories
```
.
├── Data/                 # Directory for datasets
├── MyFunctions/          # Custom functions and utilities
├── images/               # Directory containing images
├── libs/                 # Python modules for specific tasks
├── .gitignore            # Git ignore file for excluding unnecessary files
├── README.md             # Project documentation
├── recommendation.ipynb  # Notebook for the recommendation system implementation
├── clustering.ipynb      # Notebook for clustering analysis and implementation
├── algorithm.ipynb       # Notebook for the algorithmic problem
├── commandline.sh        # Bash script for Command Line Question
├── LICENSE               # MIT license for the repository
└── main.ipynb            # Main notebook combining all tasks
```
--- 


1. `README.md`: A markdown file that explains the content of the repository.

2. `main.ipynb`: A [Jupyter Notebook](link_notebook) file containing all the relevant exercises and reports belonging to the homework questions, the *Command Line Question*, and the *Algorithmic Question*.

3. ``libs/``: A folder including 4 Python modules used to solve the exercises in `main.ipynb`. The files included are:

    - `__init__.py`: A *init* file that allows us to import the modules into our Jupyter Notebook.

    - `analysis`: A Python file including a `DataHandler` class designed to handle data cleaning and feature engineering on Kaggle's [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv).

    - `recommender.py`: A Python file including a `Recommender` class designed to build a Recommendation Engine with LSH using user data obtained from Kaggle's [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv).

    - `cluster.py`: A Python file including three classes: `FAMD`, `KMeans`, and `KMeans++` designed to perform Factor Analysis of Mixed Data on Kaggle's [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv) and then perform parallelized k-Means and k-Means++ clustering using PySpark.

    - `plotter.py`: A Python file including a `Plotter` class designed to build auxiliary plots for the written report on `main.ipynb`.

4. `commandline.sh`: A bash script including the code to solve the *Command Line Question*.

5. `images/`: A folder containing a screenshot of the successful execution of the `commandline.sh` script.

6. ``.gitignore``: A predetermined `.gitignore` file that tells Git which files or folders to ignore in a Python project.

7. `LICENSE`: A file containing an MIT permissive license.

---

## Project Overview

This project explores movie recommendation systems and clustering techniques using the Kaggle [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv). It is divided into three main components:
1. **Recommendation System with Locality-Sensitive Hashing (LSH)**: Matches similar users and recommends movies based on their preferences.
2. **Clustering Movies**: Groups movies into clusters based on engineered features for better analysis.
3. **Algorithmic Question**: Solves a strategy-based game problem with an optimal solution.

---

## Dataset

The Kaggle [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv) was used for this project. It includes user ratings and movie metadata, enabling robust feature engineering and analysis.

---

## Implementation Details

### Recommendation System
1. **MinHashing**: Implemented a custom MinHash function to create user signatures.
2. **Locality-Sensitive Hashing (LSH)**: Clustered users into buckets and recommended movies based on similarity.
3. **Recommendation Logic**: Delivered personalized recommendations using weighted scoring.

### Clustering Movies
1. **Feature Engineering**: Derived multiple features, including genres, average ratings, and user tags.
2. **Clustering**:
   - Implemented K-means and K-means++ algorithms.
   - Used AI-recommended clustering methods for comparative analysis.
3. **Evaluation**: Assessed clustering quality using metrics such as Silhouette Score and WCSS.

### Algorithmic Problem
1. Developed solutions for a sequence-based game problem using recursion and dynamic programming.
2. Validated solutions with time complexity analysis and AI-assisted optimization.

---

## Results and Visualizations

### Recommendation System
- Successfully implemented a scalable recommendation engine.

### Clustering Movies
- Visualized movie groupings to uncover natural patterns.

### Algorithmic Problem
- Delivered an efficient solution validated through test cases.

---
## Important Note

If the Notebook doesn't load through Github please try all of these steps:

1. Try compiling the Notebook through its [NBViewer](mainjupiter_link).

2. Try downloading the Notebook and opening it in your local computer.

---

**Author:** Roberto Magno Mazzotta , Gabriel Pinos, xxx
