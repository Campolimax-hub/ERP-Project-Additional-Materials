import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# data reading
#data is read into the semantic similarity matrix (n×n)
semantic = pd.read_csv("D:/astudy/1ds/1ERP/dataset/semantic_similarity_matrix.csv", index_col=0)

# read in the co-authorship matrix (n×n)
coauthor = pd.read_csv("D:/astudy/1ds/1ERP/dataset/complete_coauthorship_matrix.csv", index_col=0)

# read in the author-institution mapping table
aff = pd.read_csv("D:/astudy/1ds/1ERP/dataset/node_affiliations.csv")  # the column names should be included author_id, affiliation

# node alignment
# find the common author set of the three
common_nodes = list(set(semantic.index) & set(coauthor.index) & set(aff["author_id"]))
common_nodes.sort()

semantic = semantic.loc[common_nodes, common_nodes]
coauthor = coauthor.loc[common_nodes, common_nodes]

# constructing the mechanism matrix
aff_map = dict(zip(aff["author_id"], aff["affiliation"]))
n = len(common_nodes)
institution = pd.DataFrame(0, index=common_nodes, columns=common_nodes, dtype=int)

aff_values = [aff_map.get(a, "") for a in common_nodes]
for i, ai in enumerate(aff_values):
    row = (pd.Series(aff_values) == ai).astype(int).values
    institution.iloc[i, :] = row
# remove self-loop
np.fill_diagonal(institution.values, 0)

# prepare QAP data
# expand the upper triangle matrix into a vector
def mat_to_vec(mat):
    return mat.where(np.triu(np.ones(mat.shape), k=1).astype(bool)).stack().values

y = mat_to_vec(coauthor)
X_sem = mat_to_vec(semantic)
X_inst = mat_to_vec(institution)

X = np.vstack([X_sem, X_inst]).T  # shape (dyads, 2)

# QAP regression function
def qap_regression(y, X, n_permutations=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    model = LinearRegression().fit(X, y)
    betas = model.coef_
    r2 = model.score(X, y)

    # permutation test
    perm_betas = []
    for _ in tqdm(range(n_permutations), desc="Permutations"):
        # randomly shuffle node labels -> shuffle the rows and columns of X
        perm_idx = rng.permutation(len(common_nodes))
        X_perm_sem = semantic.values[perm_idx][:, perm_idx]
        X_perm_inst = institution.values[perm_idx][:, perm_idx]

        Xp = np.vstack([mat_to_vec(pd.DataFrame(X_perm_sem)),
                        mat_to_vec(pd.DataFrame(X_perm_inst))]).T
        model_p = LinearRegression().fit(Xp, y)
        perm_betas.append(model_p.coef_)

    perm_betas = np.array(perm_betas)
    p_values = ((np.abs(perm_betas) >= np.abs(betas)).sum(axis=0) + 1) / (n_permutations + 1)

    return betas, r2, p_values

# run QAP regression
betas, r2, pvals = qap_regression(y, X, n_permutations=1000)

print("QAP Regression Results")
print("-----------------------")
print(f"R² = {r2:.3f}")
print("Variable        Beta       p-value")
print(f"Semantic      {betas[0]:.3f}     {pvals[0]:.3f}")
print(f"Institution   {betas[1]:.3f}     {pvals[1]:.3f}")

# univariate model 1: Semantics only
X_sem_only = X_sem.reshape(-1, 1)
betas_sem, r2_sem, pvals_sem = qap_regression(y, X_sem_only, n_permutations=1000)

print("\nQAP Regression - Semantic only")
print("-------------------------------")
print(f"R² = {r2_sem:.3f}")
print(f"Semantic   {betas_sem[0]:.3f}   p = {pvals_sem[0]:.3f}")

# univariate Model 2: Institutions Only
X_inst_only = X_inst.reshape(-1, 1)
betas_inst, r2_inst, pvals_inst = qap_regression(y, X_inst_only, n_permutations=1000)

print("\nQAP Regression - Institution only")
print("---------------------------------")
print(f"R² = {r2_inst:.3f}")
print(f"Institution   {betas_inst[0]:.3f}   p = {pvals_inst[0]:.3f}")

import pandas as pd
import numpy as np

# use the data prepared in the previous step
df = pd.DataFrame({
    "coauthor": y,              # whether to cooperate(0/1)
    "semantic": X_sem,          # semantic similarity
    "institution": X_inst       # is semantic similarity the same as organizational structure (0/1)
})

# calculate descriptive statistics
stats = {
    "cooperation dyads quantity": df.loc[df.coauthor==1].shape[0],
    "not cooperating dyads quantity": df.loc[df.coauthor==0].shape[0],
    "cooperation dyads semantic similarity average": df.loc[df.coauthor==1,"semantic"].mean(),
    "not cooperating dyads semantic similarity average": df.loc[df.coauthor==0,"semantic"].mean(),
    "cooperation dyads institutional proportion": df.loc[df.coauthor==1,"institution"].mean(),
    "not cooperating dyads institutional proportion": df.loc[df.coauthor==0,"institution"].mean()
}

stats_table = pd.DataFrame(stats, index=["statistical results"]).T
print(stats_table)

import networkx as nx
import matplotlib.pyplot as plt

# build a network diagram
G = nx.from_pandas_adjacency(coauthor)  # co-authoring network

# classify node colors based on institutions
institutions = [aff_map.get(node, "Unknown") for node in coauthor.index]
inst_labels = {inst:i for i,inst in enumerate(set(institutions))}
node_colors = [inst_labels[i] for i in institutions]

plt.figure(figsize=(10,10))
pos = nx.spring_layout(G, k=0.15, seed=42)

# draw nodes
nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, cmap=plt.cm.Set3)

# draw edges (with thickness weighted by semantic similarity)
edges, weights = zip(*nx.get_edge_attributes(G, "weight").items()) \
    if nx.get_edge_attributes(G, "weight") else ([],[])

# if there is no semantic information as an edge attribute, the assignment can be directly based on the semantic matrix
weights = [semantic.loc[u,v] for u,v in G.edges()]

nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color=weights, edge_cmap=plt.cm.Blues)

plt.title("Coauthorship Network (nodes=institutions, edges=semantic similarity)")
plt.axis("off")
plt.show()

from scipy.stats import pearsonr

def qap_correlation(A, B, n_permutations=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    
    # expand the upper triangular matrix into a vector
    def mat_to_vec(mat):
        return mat.where(np.triu(np.ones(mat.shape), k=1).astype(bool)).stack().values
    
    vec_A = mat_to_vec(A)
    vec_B = mat_to_vec(B)
    
    obs_corr, _ = pearsonr(vec_A, vec_B)
    
    # permutation test
    perm_corrs = []
    for _ in range(n_permutations):
        perm_idx = rng.permutation(A.shape[0])
        B_perm = B.values[perm_idx][:, perm_idx]
        vec_Bp = mat_to_vec(pd.DataFrame(B_perm))
        perm_corrs.append(pearsonr(vec_A, vec_Bp)[0])
    
    p_val = ((np.abs(perm_corrs) >= np.abs(obs_corr)).sum() + 1) / (n_permutations + 1)
    
    return obs_corr, p_val

# example: Co-authorship vs Semantics
corr_sem, p_sem = qap_correlation(coauthor, semantic, n_permutations=1000)
print("QAP Correlation (coauthor ~ semantic):", corr_sem, "p =", p_sem)

# example: Co-authorship vs Institution
corr_inst, p_inst = qap_correlation(coauthor, institution, n_permutations=1000)
print("QAP Correlation (coauthor ~ institution):", corr_inst, "p =", p_inst)

# group QAP correlation
def run_grouped_qap(aff, group_col, min_size=10, n_perm=500):
    results = []
    for group_name, group_nodes in aff.groupby(group_col):
        nodes = [n for n in common_nodes if n in group_nodes["author_id"].values]
        if len(nodes) < min_size:
            continue  # skip subgroups that are too small
        sub_co = coauthor.loc[nodes, nodes]
        sub_sem = semantic.loc[nodes, nodes]
        sub_inst = institution.loc[nodes, nodes]

        corr, p = qap_correlation(sub_co, sub_sem, n_permutations=n_perm)
        results.append({
            "group": group_name,
            "n_nodes": len(nodes),
            "corr_semantic": corr,
            "p_semantic": p
        })
    return pd.DataFrame(results)

# example: Group by institution
group_results = run_grouped_qap(aff, group_col="affiliation", min_size=10, n_perm=500)

print(group_results.head())
group_results.to_csv("qap_group_results.csv", index=False)

import seaborn as sns
import matplotlib.pyplot as plt

# semantic similarity matrix heatmap (without coordinate names)
plt.figure(figsize=(8,6))
sns.heatmap(semantic, cmap="Blues",
            cbar_kws={"label": "Semantic similarity"},
            xticklabels=False, yticklabels=False)
plt.title("Semantic Similarity Matrix (Heatmap)")
plt.show()

# co-authorship matrix heatmap (with coordinate names removed)
plt.figure(figsize=(8,6))
sns.heatmap(coauthor, cmap="Reds",
            cbar_kws={"label": "Coauthorship"},
            xticklabels=False, yticklabels=False)
plt.title("Coauthorship Matrix (Heatmap)")
plt.show()

df_plot = pd.DataFrame({
    "coauthor": y,
    "semantic": X_sem,
    "institution": X_inst
})

plt.figure(figsize=(6,6))
sns.stripplot(x="coauthor", y="semantic", data=df_plot, jitter=0.3, alpha=0.5)
plt.xticks([0,1], ["No Collaboration", "Collaboration"])
plt.ylabel("Semantic similarity")
plt.title("Semantic similarity vs Collaboration")
plt.show()

# or directly plot a scatter plot (semantic similarity vs. cooperation probability)
plt.figure(figsize=(6,6))
sns.scatterplot(x="semantic", y="coauthor", alpha=0.2)
plt.title("Scatterplot: Semantic similarity vs Coauthorship")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def plot_semantic_vs_collab(df, bins=20, frac=0.3, method="cut"):
    """
     plot semantic similarity vs. collaboration probability
    - blue dot: Cooperation probability after binning
    - red line: LOWESS smoothed curve
    
    parameter:
        df : DataFrame, include 'semantic' (semantic similarity) 和 'coauthor' (0/1 cooperation)
        bins : int, number of bins
        frac : float, LOWESS smoothing parameter (0~1, the larger, the smoother)
        method : "cut" (fixed width binning) | "qcut" (boxplot by quantile)
    """
    df = df.copy()
    df["coauthor"] = (df["coauthor"] > 0).astype(int)

    # binning
    if method == "qcut":
        df["bin"] = pd.qcut(df["semantic"], q=bins, duplicates="drop")
    else:
        df["bin"] = pd.cut(df["semantic"], bins=bins)

    # the mean value of each interval
    bin_means = df.groupby("bin").agg(
        semantic_mean=("semantic", "mean"),
        collaboration_rate=("coauthor", "mean")
    ).dropna()

    # if the number of intervals is too small, remind the user
    if len(bin_means) < 3:
        print("⚠️ after binning, the effective interval is too small. Please check the distribution of semantic similarity or increase it bins")
        print(bin_means)
        return

    # LOWESS smooth
    smoothed = sm.nonparametric.lowess(
        bin_means["collaboration_rate"],
        bin_means["semantic_mean"],
        frac=frac
    )

    # drawing
    plt.figure(figsize=(7,6))
    plt.scatter(bin_means["semantic_mean"], bin_means["collaboration_rate"],
                color="blue", s=40, label="Binned collaboration rate")
    plt.plot(smoothed[:,0], smoothed[:,1], color="red", linewidth=2, label="LOWESS fit")
    plt.xlabel("Semantic similarity")
    plt.ylabel("Probability of collaboration")
    plt.title("Semantic similarity vs Collaboration probability")
    plt.ylim(0,1)
    plt.legend()
    plt.show()

# let's first look at the distribution of semantic similarity
print(df_plot["semantic"].describe())
sns.histplot(df_plot["semantic"], bins=50)
plt.show()

# draw the main diagram
plot_semantic_vs_collab(df_plot, bins=20, frac=0.3, method="cut")

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def plot_semantic_vs_collab_subset(df, bins=20, frac=0.3, threshold=0.1, method="cut"):
    """
    draw a subsample (semantic > threshold) semantic similarity vs probability of cooperation
    
    parameter:
        df : DataFrame，include 'semantic' 和 'coauthor'
        bins : int, number of bins
        frac : float, LOWESS Smoothing parameter
        threshold : float, subsample threshold (default 0.1)
        method : "cut" (fixed width binning) | "qcut" (quantile binning)
    """
    df = df.copy()
    df["coauthor"] = (df["coauthor"] > 0).astype(int)

    # filter subsample
    df = df[df["semantic"] > threshold]
    if df.empty:
        print("⚠️ the sub-sample is empty, please reduce it threshold")
        return

    # binning
    if method == "qcut":
        df["bin"] = pd.qcut(df["semantic"], q=bins, duplicates="drop")
    else:
        df["bin"] = pd.cut(df["semantic"], bins=bins)

    # calculate the mean value of each interval
    bin_means = df.groupby("bin").agg(
        semantic_mean=("semantic","mean"),
        collaboration_rate=("coauthor","mean")
    ).dropna()

    if len(bin_means) < 3:
        print("⚠️ after binning, the effective intervals are too few, please adjust bins or threshold")
        print(bin_means)
        return

    # LOWESS smooth
    smoothed = sm.nonparametric.lowess(
        bin_means["collaboration_rate"],
        bin_means["semantic_mean"],
        frac=frac
    )

    # drawing
    plt.figure(figsize=(7,6))
    plt.scatter(bin_means["semantic_mean"], bin_means["collaboration_rate"],
                color="blue", s=40, label="Binned collaboration rate")
    plt.plot(smoothed[:,0], smoothed[:,1], color="red", linewidth=2, label="LOWESS fit")
    plt.xlabel("Semantic similarity")
    plt.ylabel("Probability of collaboration")
    plt.title(f"Semantic similarity vs Collaboration probability (semantic > {threshold})")
    plt.ylim(0,1)
    plt.legend()
    plt.show()

plot_semantic_vs_collab_subset(df_plot, bins=20, frac=0.3, threshold=0.1, method="cut")

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# define utility functions
def mat_to_vec(mat: pd.DataFrame):
    """expand the upper triangular matrix into a one-dimensional vector"""
    return mat.where(
        pd.DataFrame(np.triu(np.ones(mat.shape), k=1), index=mat.index, columns=mat.columns).astype(bool)
    ).stack().values

# read data
# you need to prepare three matrices in advance to ensure the node order is consistent(index = author_id, columns = author_id)
semantic = pd.read_csv("D:/astudy/1ds/1ERP/dataset/semantic_similarity_matrix.csv", index_col=0)
coauthor = pd.read_csv("D:/astudy/1ds/1ERP/dataset/complete_coauthorship_matrix.csv", index_col=0)
institution = pd.read_csv("D:/astudy/1ds/1ERP/dataset/qap_institution_523.csv", index_col=0)  # same institution=1, otherwise=0

# build dyad-level data
df_plot = pd.DataFrame({
    "semantic": mat_to_vec(semantic),
    "coauthor": mat_to_vec(coauthor),
    "institution": mat_to_vec(institution)
})

# dichotomy coauthor（cooperation=1，otherwise=0）
df_plot["coauthor"] = (df_plot["coauthor"] > 0).astype(int)

print(df_plot.head())

# drawing function
def plot_semantic_vs_collab_by_institution(df, bins=20, frac=0.3, threshold=0.1, method="cut"):
    """
    semantic similarity vs collaboration probability based on whether they are drawn by the same institution
    - blue dot + red line: Cross-institution
    - green dot + green line: same institution
    """
    df = df.copy()
    df = df[df["semantic"] > threshold]  # filter low similarity

    plt.figure(figsize=(7,6))

    for label, color_point, color_line in [
        ("Cross-institution", "blue", "red"),
        ("Same-institution", "green", "green")
    ]:
        sub = df[df["institution"] == (0 if label=="Cross-institution" else 1)]
        if sub.empty: 
            continue

        # binning
        if method == "qcut":
            sub["bin"] = pd.qcut(sub["semantic"], q=bins, duplicates="drop")
        else:
            sub["bin"] = pd.cut(sub["semantic"], bins=bins)

        bin_means = sub.groupby("bin").agg(
            semantic_mean=("semantic","mean"),
            collaboration_rate=("coauthor","mean")
        ).dropna()

        if len(bin_means) < 3:
            continue

        # LOWESS smooth
        smoothed = sm.nonparametric.lowess(
            bin_means["collaboration_rate"],
            bin_means["semantic_mean"],
            frac=frac
        )

        # drawing points and lines
        plt.scatter(bin_means["semantic_mean"], bin_means["collaboration_rate"],
                    color=color_point, s=40, label=f"{label} (binned rate)")
        plt.plot(smoothed[:,0], smoothed[:,1],
                 color=color_line, linewidth=2, label=f"{label} (LOWESS fit)")

    plt.xlabel("Semantic similarity")
    plt.ylabel("Probability of collaboration")
    plt.title(f"Semantic similarity vs Collaboration probability (>{threshold})")
    plt.ylim(0,1)
    plt.legend()
    plt.show()

# call the drawing function.
plot_semantic_vs_collab_by_institution(df_plot, bins=20, frac=0.3, threshold=0.1, method="cut")

