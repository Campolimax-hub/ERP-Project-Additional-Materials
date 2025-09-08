# ERP-Project-Additional-Materials
This repository contains all code and materials required to reproduce the analysis for my MSc Data Science (Social Research) ERP project: “Cultural Homophily and Academic Collaboration: A Socio-Semantic Network Analysis.”

****Repository Structure****


**code1.py**

Cleans the raw OpenAlex article data

Extracts authors and affiliations

Builds the full co-authorship matrix and exports key mapping files

Generates .csv and .vna files for network construction

**code2.py**

Computes semantic similarity between authors using Sentence-Transformers embeddings

Builds the semantic similarity matrix

Integrates co-authorship, semantic, and institutional affiliation into a multilayer network

Outputs edge lists and node attributes for visualization (e.g., NetDraw, Gephi)

**code3.py**

Performs QAP correlation and regression analysis on co-authorship vs. semantic and institutional networks

Provides descriptive statistics on dyads

Generates visualizations (heatmaps, scatter plots, LOWESS curves)

Exports subgroup-level QAP results (by institution)


**Data Requirements**

The scripts assume access to a dataset of academic articles (collected from OpenAlex) with the following files prepared:

raw-data-articles.csv – **raw article metadata**


**Due to data policy, raw datasets are not included here. Users should obtain OpenAlex data independently.**


**Dependencies**

Install the required Python packages:

pip install pandas numpy scikit-learn tqdm sentence-transformers torch torchvision torchaudio networkx seaborn statsmodels matplotlib

**How to Run**

Run code1.py to clean raw data and build the co-authorship matrix.

Run code2.py to compute semantic similarity and construct multilayer networks.

Run code3.py to perform QAP regression, descriptive analysis, and generate figures.

Each script will output .csv and .png files into the dataset folder.



**Outputs**

Matrices: semantic_similarity_matrix.csv, complete_coauthorship_matrix.csv

Edge/Node files: semantic_similarity_edges.csv, node_affiliations.csv

Figures: heatmaps, scatter plots, final multilayer network visualization

Statistical results: subgroup-level QAP correlations and regression outputs
