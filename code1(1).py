import pandas as pd

#read dataset
raw_df = pd.read_csv("D:/astudy/1ds/1ERP/dataset/raw-data-articles.csv", encoding="ISO-8859-1", low_memory=False)

#keep the important information
keep_columns = [
    'id',
    'title',
    'publication_year',
    'authorships.author.id',
    'authorships.author.display_name',
    'authorships.raw_affiliation_strings',
    'authorships.author.orcid',
    'type',
    'biblio.volume',
    'biblio.issue',
    'biblio.first_page',
    'biblio.last_page',
    'host_venue.display_name',
    'host_venue.issn_l',
    'host_venue.publisher',
    'host_venue.is_oa',
    'host_venue.is_in_doaj',
    'host_venue.source_id'
    ]

#keep the important information
filtered_df = raw_df[[col for col in keep_columns if col in raw_df.columns]]

#export cleaned dataset
filtered_df.to_csv("D:/astudy/1ds/1ERP/dataset/cleaned_articles_data.csv", index=False, encoding="ISO-8859-1")

import pandas as pd

#read the cleaned dataset
df = pd.read_csv("D:/astudy/1ds/1ERP/dataset/cleaned_articles_data.csv", encoding="ISO-8859-1")

#read author pairs based on ID
def extract_author_pairs(row):
    author_ids = str(row['authorships.author.id']).split('|')
    pairs = []
    if len(author_ids) > 1:
        for i in range(len(author_ids)):
            for j in range(i + 1, len(author_ids)):
                a1 = author_ids[i].strip()
                a2 = author_ids[j].strip()
                if a1 and a2:
                    pairs.append(tuple(sorted([a1, a2])))
    return pairs

df['author_pairs'] = df.apply(extract_author_pairs, axis=1)
pairs_df = df[['id', 'authorships.author.id', 'authorships.author.display_name', 'author_pairs']].explode('author_pairs').dropna()

#split Author 1 and Author 2
pairs_df[['Author 1 ID', 'Author 2 ID']] = pd.DataFrame(pairs_df['author_pairs'].tolist(), index=pairs_df.index)

#build map of ID to Name and add author's name
id_to_name = {}
for _, row in df.iterrows():
    ids = str(row['authorships.author.id']).split('|')
    names = str(row['authorships.author.display_name']).split('|')
    for aid, name in zip(ids, names):
        id_to_name[aid.strip()] = name.strip()

pairs_df['Author 1 Name'] = pairs_df['Author 1 ID'].map(id_to_name)
pairs_df['Author 2 Name'] = pairs_df['Author 2 ID'].map(id_to_name)

#organize output structure
pairs_df['Weight'] = 1  # the co-authors in each article are assigned a default weight of 1
pairs_df.rename(columns={'id': 'Paper'}, inplace=True)
final_df = pairs_df[['Paper', 'Author 1 ID', 'Author 2 ID', 'Weight', 'Author 1 Name', 'Author 2 Name']]

#export CSV file
final_df.to_csv("D:/astudy/1ds/1ERP/dataset/article_author_pairs_with_ids.csv", index=False)

# extract the list from the previous dataset(all the author combinations in each article)
from itertools import combinations
from collections import defaultdict

# extract the author list from all articles
article_authors = df['authorships.author.id'].dropna().apply(lambda x: x.split('|'))

# coun the number of co-authorships for each pair of authors
coauthor_counts = defaultdict(int)
for authors in article_authors:
    if len(authors) < 2:
        continue
    for a1, a2 in combinations(sorted(authors), 2):
        coauthor_counts[(a1, a2)] += 1

# extract all authors IDs
all_authors = sorted(set([a for pair in coauthor_counts for a in pair]))

# create a co-authorship matrix(DataFrame)
coauthor_matrix_full = pd.DataFrame(0, index=all_authors, columns=all_authors)

# enter the number of co-authorships(symmetric matrix)
for (a1, a2), count in coauthor_counts.items():
    coauthor_matrix_full.at[a1, a2] = count
    coauthor_matrix_full.at[a2, a1] = count

# show some rows
import ace_tools as tools; tools.display_dataframe_to_user(name="Full Coauthorship Matrix", dataframe=coauthor_matrix_full.head(10))

# save as csv file
csv_output_path = "D:/astudy/1ds/1ERP/dataset/full_coauthorship_matrix.csv"
coauthor_matrix_full.to_csv(csv_output_path)

csv_output_path

# obtain all author IDs that appear in the organisation mapping table(including those who do not participate in co-authors)
all_authors_full_set = sorted(set(final_full_df['author_id']))

# create an empty co-author matrix for all authors, and the default number of co-authors is 0
full_matrix = pd.DataFrame(0, index=all_authors_full_set, columns=all_authors_full_set)

# fill in the existing number of co-authors into the new matrix(aligned by index)
for a1 in coauthor_matrix_full.index:
    for a2 in coauthor_matrix_full.columns:
        full_matrix.at[a1, a2] = coauthor_matrix_full.at[a1, a2]

# keep the most frequent organization ID for each author
author_primary_inst = (
    final_full_df
    .groupby(['author_id', 'institution_id'])
    .size()
    .reset_index(name='count')
    .sort_values(['author_id', 'count'], ascending=[True, False])
    .drop_duplicates(subset=['author_id'])
    .reset_index(drop=True)
    [['author_id', 'institution_id']]
)

#show outcome
import ace_tools as tools; tools.display_dataframe_to_user(name="Primary Author-Institution Mapping", dataframe=author_primary_inst)

# export the unique author-organization correspondence as a CSV file
primary_inst_path = "D:/astudy/1ds/1ERP/dataset/primary_author_institution_mapping.csv"
author_primary_inst.to_csv(primary_inst_path, index=False)

primary_inst_path

# read the uploaded author's file
pairs_file_path = "D:/astudy/1ds/1ERP/dataset/article_author_pairs_with_ids.csv"
pairs_df = pd.read_csv(pairs_file_path)

# preview the listing
pairs_df.columns.tolist()

# count the number of co-authors between each pair of author IDs
coauthor_pair_counts = (
    pairs_df.groupby(['Author 1 ID', 'Author 2 ID'])
    .size()
    .reset_index(name='Coauthorship Count')
)

# show outcome
import ace_tools as tools; tools.display_dataframe_to_user(name="Coauthorship Pair Counts", dataframe=coauthor_pair_counts)

# export the co-authored pair and the number of co-authors as a CSV file
coauthor_count_path = "D:/astudy/1ds/1ERP/dataset/coauthorship_pair_counts.csv"
coauthor_pair_counts.to_csv(coauthor_count_path, index=False)

coauthor_count_path

# load the author-organization master mapping table, which is used to generate node attribute information in VNA
inst_attr_path = "D:/astudy/1ds/1ERP/dataset/primary_author_institution.csv"
inst_attr_df = pd.read_csv(inst_attr_path)

# preparation of node information
inst_attr_df['Label'] = inst_attr_df['author_id'].apply(lambda x: x.split('/')[-1])  # use the last paragraph of ID as a label
inst_attr_df['ID'] = inst_attr_df['author_id']
inst_attr_df['institution'] = inst_attr_df['institution_id']

nodes_df = inst_attr_df[['ID', 'Label', 'institution']]

# edge information preparation(previous generated co-authorship pairs)
edges_df = coauthor_pair_counts.rename(
    columns={
        'Author 1 ID': 'from',
        'Author 2 ID': 'to',
        'Coauthorship Count': 'strength'
    }
)

# construct .vna file content
vna_lines = []

# node part
vna_lines.append("*node data")
vna_lines.append("ID\tLabel\tinstitution")
for _, row in nodes_df.iterrows():
    vna_lines.append(f"{row['ID']}\t{row['Label']}\t{row['institution']}")

# edge part
vna_lines.append("*tie data")
vna_lines.append("from\tto\tstrength")
for _, row in edges_df.iterrows():
    vna_lines.append(f"{row['from']}\t{row['to']}\t{row['strength']}")

# write the file
vna_path = "D:/astudy/1ds/1ERP/dataset/coauthorship_network.vna"
with open(vna_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(vna_lines))

vna_path

import pandas as pd
import networkx as nx

# read the data uploaded by the user
edges_path = "D:/astudy/1ds/1ERP/dataset/coauthorship_pair_counts.csv"
nodes_path = "D:/astudy/1ds/1ERP/dataset/primary_author_institution.csv"

edges_df = pd.read_csv(edges_path)
nodes_df = pd.read_csv(nodes_path)

# construct diagram
G = nx.Graph()
for _, row in edges_df.iterrows():
    G.add_edge(row['Author 1 ID'], row['Author 2 ID'], weight=row['Coauthorship Count'])

# network basic indicators
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
density = nx.density(G)
degrees = dict(G.degree())
avg_degree = sum(degrees.values()) / num_nodes
max_degree = max(degrees.values())

# number of connected sub-charts
num_components = nx.number_connected_components(G)

# maximum connected sub-chart
largest_cc = max(nx.connected_components(G), key=len)
G_lcc = G.subgraph(largest_cc)

# average path length and diameter(in the larges sub-chart)
avg_path_length = nx.average_shortest_path_length(G_lcc)
diameter = nx.diameter(G_lcc)

# summary result
network_stats = {
    "Node number(author)": num_nodes,
    "Edge number(co-author pair)": num_edges,
    "Network density": round(density, 4),
    "Average degree": round(avg_degree, 2),
    "Maximum degree": max_degree,
    "Number of connected sub-charts": num_components,
    "Average path degree": round(avg_path_length, 3),
    "Network diameter": diameter
}

network_stats

