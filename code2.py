pip install sentence-transformers pandas scikit-learn tqdm
!pip install torch torchvision torchaudio --index-url https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
import torch
print(torch.__version__)

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

#load dataset
articles_df = pd.read_csv("D:/astudy/1ds/1ERP/dataset/cleaned_articles_data.csv", encoding='ISO-8859-1')
pairs_df = pd.read_csv("D:/astudy/1ds/1ERP/dataset/article_author_pairs_with_ids.csv", encoding='ISO-8859-1')

#clean the author-article paring relationship ===
article_ids = articles_df['id']
author_id_lists = articles_df['authorships.author.id'].str.split('|')

author_article_pairs = []
for idx, authors in enumerate(author_id_lists):
    article_id = article_ids.iloc[idx]
    for author in authors:
        author_article_pairs.append({'author_id': author.strip(), 'article_id': article_id.strip()})

author_article_df = pd.DataFrame(author_article_pairs)

#build the article text and generate embedding
articles_df['abstract'] = articles_df['abstract'].fillna('')
articles_df['title'] = articles_df['title'].fillna('')
articles_df['text'] = articles_df['title'] + '. ' + articles_df['abstract']

model = SentenceTransformer('all-MiniLM-L6-v2')
articles_df['embedding'] = articles_df['text'].apply(lambda x: model.encode(x))

article_embeddings = dict(zip(articles_df['id'], articles_df['embedding']))
author_to_articles = author_article_df.groupby('author_id')['article_id'].apply(set).to_dict()
author_pairs = pairs_df[['Author 1 ID', 'Author 2 ID']].dropna().drop_duplicates()

#calculate the sematic similarity
similarity_results = []
for idx, row in tqdm(author_pairs.iterrows(), total=len(author_pairs)):
    a1, a2 = row['Author 1 ID'].strip(), row['Author 2 ID'].strip()
    articles_a1 = author_to_articles.get(a1, set())
    articles_a2 = author_to_articles.get(a2, set())
    shared = articles_a1 & articles_a2

    articles_a1_clean = articles_a1 - shared
    articles_a2_clean = articles_a2 - shared

    vecs_a1 = [article_embeddings[aid] for aid in articles_a1_clean if aid in article_embeddings]
    vecs_a2 = [article_embeddings[aid] for aid in articles_a2_clean if aid in article_embeddings]

    if vecs_a1 and vecs_a2:
        mean_a1 = np.mean(vecs_a1, axis=0).reshape(1, -1)
        mean_a2 = np.mean(vecs_a2, axis=0).reshape(1, -1)
        sim = cosine_similarity(mean_a1, mean_a2)[0][0]
        similarity_results.append({
            'Author 1 ID': a1,
            'Author 2 ID': a2,
            'Semantic Similarity': sim
        })

semantic_df = pd.DataFrame(similarity_results)
semantic_df.to_csv("author_pair_semantic_similarity.csv", index=False)

import pandas as pd

#load the co-authored adjacency matrix and extract the author's ID list
coauthor_matrix = pd.read_csv("D:/astudy/1ds/1ERP/dataset/complete_coauthorship_matrix.csv", index_col=0)
author_ids = list(coauthor_matrix.index)

#load the semantic similarity table
semantic_df = pd.read_csv("author_pair_semantic_similarity.csv")

#keep the author only in the co-author matrix
semantic_filtered = semantic_df[
    (semantic_df['Author 1 ID'].isin(author_ids)) &
    (semantic_df['Author 2 ID'].isin(author_ids))
]

#save as the edge attribute format supported by NetDraw(original author ID)
semantic_filtered[['Author 1 ID', 'Author 2 ID', 'Semantic Similarity']].to_csv(
    "semantic_similarity_edges.csv", index=False
)
print("✅ Generated semantic_similarity_edges.csv(using the original author ID)")

# extract author and institutional information from article data 
articles_df = pd.read_csv("D:/astudy/1ds/1ERP/dataset/cleaned_articles_data.csv", encoding='ISO-8859-1')
author_id_lists = articles_df['authorships.author.id'].str.split('|')
affil_lists = articles_df['authorships.raw_affiliation_strings'].str.split('|')

author_affil_pairs = []
for idx, authors in enumerate(author_id_lists):
    affils = affil_lists.iloc[idx] if idx < len(affil_lists) else []
    for i, author in enumerate(authors):
        affil = affils[i].strip() if i < len(affils) else None
        author_affil_pairs.append({'author_id': author.strip(), 'affiliation': affil})

author_affil_df = pd.DataFrame(author_affil_pairs).dropna().drop_duplicates()

#only keep the authors who have appeared in the co-author matrix
author_affil_filtered = author_affil_df[author_affil_df['author_id'].isin(author_ids)]

#save the node attribute file(original ID)
author_affil_filtered[['author_id', 'affiliation']].to_csv(
    "node_affiliations.csv", index=False
)
print("✅ Generated node_affiliations.csv(using the original author ID)")

import pandas as pd
import numpy as np

#load semantic data
df = pd.read_csv("author_pair_semantic_similarity.csv")

#Get all author IDs and build an empty matrix
authors = sorted(set(df['Author 1 ID']) | set(df['Author 2 ID']))
matrix = pd.DataFrame(0.0, index=authors, columns=authors)

#fill in semantic similarity(smmetic matrix )
for _, row in df.iterrows():
    a1, a2, sim = row['Author 1 ID'], row['Author 2 ID'], row['Semantic Similarity']
    matrix.at[a1, a2] = sim
    matrix.at[a2, a1] = sim

#save as UCINET importable .csv adjacency matrix
matrix.to_csv("semantic_similarity_matrix.csv")
print("✅ 已生成 semantic_similarity_matrix.csv")

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# threshold setting
COAUTHOR_THRESHOLD = 2
SEMANTIC_THRESHOLD = 0.3

#retrieve data
coauthor_matrix = pd.read_csv("D:/astudy/1ds/1ERP/dataset/complete_coauthorship_matrix.csv", index_col=0)
semantic_matrix = pd.read_csv("semantic_similarity_matrix.csv", index_col=0)
node_affiliations = pd.read_csv("node_affiliations.csv")

#create a diagram
G = nx.Graph()

#add a node
for _, row in node_affiliations.iterrows():
    G.add_node(row['author_id'])

#add a co-authored edge
for i in coauthor_matrix.index:
    for j in coauthor_matrix.columns:
        weight = coauthor_matrix.loc[i, j]
        if weight >= COAUTHOR_THRESHOLD:
            G.add_edge(i, j, coauthor_weight=weight, semantic_weight=0.0)

#add semantic similar edges
for i in semantic_matrix.index:
    for j in semantic_matrix.columns:
        sim = semantic_matrix.loc[i, j]
        if sim >= SEMANTIC_THRESHOLD:
            if G.has_edge(i, j):
                G[i][j]['semantic_weight'] = sim
            else:
                G.add_edge(i, j, coauthor_weight=0, semantic_weight=sim)

#edge classification
edges_co = []
edges_sem = []
edges_both = []

for u, v, d in G.edges(data=True):
    has_co = d.get('coauthor_weight', 0) >= COAUTHOR_THRESHOLD
    has_sem = d.get('semantic_weight', 0) >= SEMANTIC_THRESHOLD
    if has_co and has_sem:
        edges_both.append((u, v, d))
    elif has_co:
        edges_co.append((u, v, d))
    elif has_sem:
        edges_sem.append((u, v, d))

# use Kamada-Kawai layout
pos = nx.kamada_kawai_layout(G)

#start drawing
plt.figure(figsize=(16, 12))

# draw the edge
nx.draw_networkx_edges(
    G, pos,
    edgelist=[(u, v) for u, v, _ in edges_co],
    width=[d['coauthor_weight'] for _, _, d in edges_co],
    edge_color='gray',
    alpha=0.4
)
nx.draw_networkx_edges(
    G, pos,
    edgelist=[(u, v) for u, v, _ in edges_sem],
    width=[d['semantic_weight'] * 5 for _, _, d in edges_sem],
    edge_color='blue',
    alpha=0.3,
    style='dashed'
)
nx.draw_networkx_edges(
    G, pos,
    edgelist=[(u, v) for u, v, _ in edges_both],
    width=[(d['coauthor_weight'] + d['semantic_weight'] * 5) / 2 for _, _, d in edges_both],
    edge_color='red',
    alpha=0.7
)

#draw node(unified black)
nx.draw_networkx_nodes(G, pos, node_color='black', node_size=50)

# add the edge legend
legend_elements = [
    mlines.Line2D([], [], color='gray', lw=2, label='Coauthor only'),
    mlines.Line2D([], [], color='blue', lw=2, linestyle='--', label='Semantic only'),
    mlines.Line2D([], [], color='red', lw=2, label='Both')
]
plt.legend(handles=legend_elements, loc='lower left', fontsize='small', frameon=True, title='Edge Types')

# clean up images
plt.axis('off')
plt.tight_layout()

#save the image
plt.savefig("final_multilayer_network.png", dpi=300)
plt.show()

import pandas as pd

#threshold setting
COAUTHOR_THRESHOLD = 2
SEMANTIC_THRESHOLD = 0.3

#read data files
coauthor_matrix = pd.read_csv("D:/astudy/1ds/1ERP/dataset/complete_coauthorship_matrix.csv", index_col=0)
semantic_matrix = pd.read_csv("semantic_similarity_matrix.csv", index_col=0)
node_affiliations = pd.read_csv("node_affiliations.csv")

#create a mapping table from the author to the organisation
affiliation_dict = node_affiliations.set_index('author_id')['affiliation'].to_dict()

#generate a list of overlapping edges
overlapping_edges = []
for i in coauthor_matrix.index:
    for j in coauthor_matrix.columns:
        if i < j:  #avoid repetition
            coauth = coauthor_matrix.loc[i, j]
            semantic = semantic_matrix.loc[i, j] if i in semantic_matrix.index and j in semantic_matrix.columns else 0
            if coauth >= COAUTHOR_THRESHOLD and semantic >= SEMANTIC_THRESHOLD:
                overlapping_edges.append({
                    'author_A': i,
                    'author_B': j,
                    'coauthor_weight': coauth,
                    'semantic_similarity': semantic,
                    'affiliation_A': affiliation_dict.get(i, 'Unknown'),
                    'affiliation_B': affiliation_dict.get(j, 'Unknown'),
                    'same_affiliation': int(affiliation_dict.get(i) == affiliation_dict.get(j))
                })

# consort to DataFrame and export
overlap_df = pd.DataFrame(overlapping_edges)
overlap_df.to_csv("overlapping_dyads.csv", index=False)

print("✅ Overlapping dyads data exported to 'overlapping_dyads.csv'")

# Load the overlapping dyads file
overlap_df = pd.read_csv("overlapping_dyads.csv")

# Basic descriptive statistics
summary_stats = overlap_df[['coauthor_weight', 'semantic_similarity']].describe()

# Proportion of dyads with same affiliation
same_aff_count = overlap_df['same_affiliation'].sum()
total_dyads = len(overlap_df)
same_aff_ratio = same_aff_count / total_dyads

summary_stats_result = summary_stats.copy()
summary_stats_result.loc['same_affiliation_ratio'] = [same_aff_ratio, None]

summary_stats_result
