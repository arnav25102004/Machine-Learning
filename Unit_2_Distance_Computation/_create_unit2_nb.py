import json

cells = []

# Title & Objective
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": (
        "# Unit 2 – Pairwise Distance Matrix Computation\n"
        "### Student: 2547115 | Subject: Machine Learning\n"
        "\n"
        "---\n"
        "\n"
        "## Q2. Demonstrate distance matrix computation using any 2 distance calculation methods.\n"
        "\n"
        "### Objective\n"
        "We will compute the pairwise distance matrix of the **Lab 2 Student Survey Dataset** using:\n"
        "1. **Euclidean Distance** (`metric='euclidean'`)\n"
        "2. **Manhattan Distance** (`metric='cityblock'`)\n"
        "\n"
        "We will preprocess the numeric data, compute the matrices using `sklearn.metrics.pairwise_distances`, visualize them with simple heatmaps, and compare them side-by-side."
    )
})

# Step 1: Imports
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": "### Step 1: Import Libraries"
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sklearn.metrics.pairwise import pairwise_distances\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "\n"
        "print('Libraries loaded successfully!')"
    )
})

# Step 2: Loading Data
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": "### Step 2: Load the Lab 2 Survey Dataset"
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "df = pd.read_excel('../LAB_2/lab2 data.xlsx')\n"
        "print('Original dataset shape:', df.shape)\n"
        "df.head()"
    )
})

# Step 3: Simple Preprocessing
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": "### Step 3: Simple Preprocessing & Feature Selection"
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "# 1. Select the numeric columns of interest\n"
        "features = [\n"
        "    'Rate your contribution towards extra curricular activities',\n"
        "    'Rate your technical competencies',\n"
        "    'What are your package expectations (LPA)',\n"
        "    'your CIA % of last semester',\n"
        "    'your GPA of last semester',\n"
        "    'Your maximum attendance % till last semester'\n"
        "]\n"
        "\n"
        "# 2. Clean and convert to numeric values\n"
        "df_clean = df[features].copy()\n"
        "for col in df_clean.columns:\n"
        "    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')\n"
        "\n"
        "# Fix decimal entries in attendance (e.g., 0.95 -> 95)\n"
        "df_clean['Your maximum attendance % till last semester'] = df_clean['Your maximum attendance % till last semester'].apply(\n"
        "    lambda x: x * 100 if pd.notna(x) and x < 1.5 else x\n"
        ")\n"
        "\n"
        "# 3. Drop rows with missing values\n"
        "df_clean = df_clean.dropna().reset_index(drop=True)\n"
        "print('Cleaned dataset shape:', df_clean.shape)"
    )
})

# Step 4: Scale Features
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": (
        "### Step 4: Standardize the Features\n"
        "Scaling is required because metrics like Package Expectation and CIA % have different scales. StandardScaler transforms each feature to have a mean of 0 and variance of 1."
    )
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "scaler = StandardScaler()\n"
        "X = scaler.fit_transform(df_clean)\n"
        "print('Data scaled successfully.')"
    )
})

# Step 5: Compute Euclidean Distance
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": (
        "### Step 5: Method 1 — Compute Euclidean Distance Matrix\n"
        "Euclidean distance measures the straight-line distance between two points in multidimensional space:\n"
        "$$d(x, y) = \\sqrt{\\sum_{i=1}^{n} (x_i - y_i)^2}$$"
    )
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "D_euclidean = pairwise_distances(X, metric='euclidean')\n"
        "print('Euclidean distance matrix shape:', D_euclidean.shape)\n"
        "print('Example matrix (first 5x5 students):')\n"
        "print(np.round(D_euclidean[:5, :5], 3))"
    )
})

# Step 6: Compute Manhattan Distance
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": (
        "### Step 6: Method 2 — Compute Manhattan Distance Matrix\n"
        "Manhattan distance is the sum of the absolute differences across all dimensions:\n"
        "$$d(x, y) = \\sum_{i=1}^{n} |x_i - y_i|$$"
    )
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "D_manhattan = pairwise_distances(X, metric='cityblock')\n"
        "print('Manhattan distance matrix shape:', D_manhattan.shape)\n"
        "print('Example matrix (first 5x5 students):')\n"
        "print(np.round(D_manhattan[:5, :5], 3))"
    )
})

# Step 7: Plot Heatmaps
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": "### Step 7: Visualize Both Distance Matrices Side-by-Side"
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "fig, axes = plt.subplots(1, 2, figsize=(16, 7))\n"
        "\n"
        "# We visualize the first 30 students to make the heatmap clear and readable\n"
        "N = 30\n"
        "\n"
        "# Euclidean Heatmap\n"
        "sns.heatmap(D_euclidean[:N, :N], ax=axes[0], cmap='Blues', annot=True, fmt='.1f', annot_kws={'size': 8}, square=True)\n"
        "axes[0].set_title('Euclidean Distance Matrix (First 30 Students)')\n"
        "axes[0].set_xlabel('Student Index')\n"
        "axes[0].set_ylabel('Student Index')\n"
        "\n"
        "# Manhattan Heatmap\n"
        "sns.heatmap(D_manhattan[:N, :N], ax=axes[1], cmap='Oranges', annot=True, fmt='.1f', annot_kws={'size': 8}, square=True)\n"
        "axes[1].set_title('Manhattan Distance Matrix (First 30 Students)')\n"
        "axes[1].set_xlabel('Student Index')\n"
        "axes[1].set_ylabel('Student Index')\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
})

# Step 8: Compare metrics
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": "### Step 8: Comparison & Discussion"
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": (
        "# Extract upper triangles to compare unique pairs\n"
        "euc_tri = D_euclidean[np.triu_indices_from(D_euclidean, k=1)]\n"
        "man_tri = D_manhattan[np.triu_indices_from(D_manhattan, k=1)]\n"
        "\n"
        "print('--- Statistical Summary ---')\n"
        "print(f'Euclidean Distances - Min: {euc_tri.min():.2f}, Max: {euc_tri.max():.2f}, Mean: {euc_tri.mean():.2f}')\n"
        "print(f'Manhattan Distances - Min: {man_tri.min():.2f}, Max: {man_tri.max():.2f}, Mean: {man_tri.mean():.2f}')\n"
        "\n"
        "# Correlation\n"
        "correlation = np.corrcoef(euc_tri, man_tri)[0, 1]\n"
        "print(f'Pearson Correlation between the two distance types: {correlation:.4f}')\n"
        "print('High correlation suggests that both distance methods preserve similar geometric patterns in this dataset.')"
    )
})

# Save notebook
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = r'm:\Machine Learning\Unit_2_Distance_Computation\2547115_Unit2_DistanceMatrix.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Simple notebook written!')
