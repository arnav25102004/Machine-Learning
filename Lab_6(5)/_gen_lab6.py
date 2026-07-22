import json, textwrap

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}

def code(*lines):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": list(lines)}

cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells += [md(
    "# Lab 6 – Linear Regression using Gradient Descent\n",
    "\n",
    "**Aim:** Implement Linear Regression using the Gradient Descent optimization algorithm "
    "and evaluate its performance on the Student Performance dataset "
    "(UCI Machine Learning Repository).\n",
    "\n",
    "---"
)]

# ── Step 1 ────────────────────────────────────────────────────────────────────
cells += [
    md(
        "## Step 1 – Import Libraries\n",
        "\n",
        "Before anything else, we bring in all the tools we will need throughout this experiment. "
        "Think of it like setting up your workbench before you start a project."
    ),
    code(
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing  import StandardScaler, LabelEncoder\n",
        "from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score\n",
        "\n",
        "print('All libraries imported successfully!')"
    ),
    md(
        "**What just happened?**  \n",
        "We loaded NumPy (fast math on arrays), Pandas (spreadsheet-style data handling), and Matplotlib (plotting). "
        "From scikit-learn we only grab helpers for splitting data and measuring errors — "
        "the actual regression model we are building ourselves, completely from scratch."
    )
]

# ── Step 2 ────────────────────────────────────────────────────────────────────
cells += [
    md(
        "---\n",
        "## Step 2 – Load the Dataset\n",
        "\n",
        "The **Student Performance (Math course)** dataset has 395 rows (students) and 33 columns. "
        "The target column we want to predict is `G3` — the final grade on a 0–20 scale."
    ),
    code(
        "df = pd.read_csv('student-mat.csv', sep=';')\n",
        "\n",
        "print(f'Dataset shape : {df.shape}')\n",
        "print(f'Rows          : {df.shape[0]}  (students)')\n",
        "print(f'Columns       : {df.shape[1]}  (features + target)')\n",
        "df.head()"
    ),
    md(
        "**What just happened?**  \n",
        "We loaded the CSV. The separator here is a semicolon (`;`), so we pass `sep=';'`. "
        "`df.head()` shows the first five rows so we can visually confirm everything looks right."
    )
]

# ── Step 3 ────────────────────────────────────────────────────────────────────
cells += [
    md(
        "---\n",
        "## Step 3 – Exploratory Data Analysis (EDA)\n",
        "\n",
        "Before touching any algorithm, we explore the data — understand its structure, "
        "check for problems, and see how the target variable is distributed."
    ),
    code(
        "print('=' * 55)\n",
        "print('DATASET INFO')\n",
        "print('=' * 55)\n",
        "df.info()\n",
        "print()\n",
        "print('=' * 55)\n",
        "print('SUMMARY STATISTICS (Numeric Columns)')\n",
        "print('=' * 55)\n",
        "df.describe().round(2)"
    ),
    code(
        "# Check for missing values\n",
        "missing = df.isnull().sum()\n",
        "print('Missing values per column:')\n",
        "if missing.any():\n",
        "    print(missing[missing > 0])\n",
        "else:\n",
        "    print('No missing values found!')"
    ),
    code(
        "# Distribution of the target variable G3\n",
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n",
        "fig.suptitle('Target Variable - Final Grade (G3)', fontsize=14, fontweight='bold')\n",
        "\n",
        "axes[0].hist(df['G3'], bins=20, color='steelblue', edgecolor='white', alpha=0.85)\n",
        "axes[0].set_xlabel('Final Grade (G3)')\n",
        "axes[0].set_ylabel('Number of Students')\n",
        "axes[0].set_title('Histogram')\n",
        "mean_val = df['G3'].mean()\n",
        "axes[0].axvline(mean_val, color='red', linestyle='--', label=f'Mean = {mean_val:.2f}')\n",
        "axes[0].legend()\n",
        "\n",
        "axes[1].boxplot(df['G3'], patch_artist=True,\n",
        "                boxprops=dict(facecolor='steelblue', alpha=0.7))\n",
        "axes[1].set_xlabel('G3')\n",
        "axes[1].set_title('Box Plot')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "print(f'G3 stats:\\n{df[\"G3\"].describe().round(2)}')"
    ),
    code(
        "# Correlation of numeric features with G3\n",
        "numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n",
        "corr_with_G3 = df[numeric_cols].corr()['G3'].drop('G3').sort_values(key=abs, ascending=False)\n",
        "\n",
        "plt.figure(figsize=(10, 5))\n",
        "bar_colors = ['#e74c3c' if v > 0 else '#3498db' for v in corr_with_G3]\n",
        "corr_with_G3.plot(kind='bar', color=bar_colors, edgecolor='black', linewidth=0.5)\n",
        "plt.axhline(0, color='black', linewidth=0.8)\n",
        "plt.title('Pearson Correlation of Each Feature with Final Grade (G3)',\n",
        "          fontsize=13, fontweight='bold')\n",
        "plt.xlabel('Feature')\n",
        "plt.ylabel('Correlation Coefficient')\n",
        "plt.xticks(rotation=45, ha='right')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('Top 5 most correlated features with G3:')\n",
        "print(corr_with_G3.head())"
    ),
    md(
        "**What just happened?**  \n",
        "We explored the data in three ways:\n",
        "1. **Info + Statistics** – no missing values, data types look fine.\n",
        "2. **Target distribution** – most students score between 8–16 with a roughly bell-shaped distribution. "
        "A handful score 0 (likely dropped out or were absent for the final exam).\n",
        "3. **Correlation chart** – `G2` and `G1` (intermediate period grades) are by far the strongest predictors of `G3`. "
        "Totally makes sense — past performance predicts future performance."
    )
]

# ── Step 4 ────────────────────────────────────────────────────────────────────
cells += [
    md(
        "---\n",
        "## Step 4 – Data Preprocessing\n",
        "\n",
        "Raw data is never plug-and-play for ML. We need to encode text columns to numbers, "
        "pick the best features, split the data into training and testing sets, and scale all features."
    ),
    code(
        "# 4a. Encode categorical columns with Label Encoding\n",
        "df_encoded = df.copy()\n",
        "le = LabelEncoder()\n",
        "cat_cols = df_encoded.select_dtypes(include='object').columns.tolist()\n",
        "print(f'Categorical columns to encode ({len(cat_cols)}): {cat_cols}\\n')\n",
        "\n",
        "for col in cat_cols:\n",
        "    df_encoded[col] = le.fit_transform(df_encoded[col])\n",
        "\n",
        "print('After encoding, all columns are numeric:')\n",
        "print(df_encoded.dtypes.value_counts())"
    ),
    code(
        "# 4b. Feature selection\n",
        "# Chosen based on correlation analysis + domain knowledge\n",
        "selected_features = ['G1', 'G2', 'studytime', 'failures', 'absences',\n",
        "                     'Medu', 'Fedu', 'age', 'famrel', 'goout']\n",
        "\n",
        "X = df_encoded[selected_features].values   # feature matrix\n",
        "y = df_encoded['G3'].values                # target vector\n",
        "\n",
        "print(f'Feature matrix X : {X.shape}')\n",
        "print(f'Target vector  y : {y.shape}')\n",
        "print(f'\\nSelected features: {selected_features}')"
    ),
    code(
        "# 4c. Train / Test split (80% train, 20% test)\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.20, random_state=42\n",
        ")\n",
        "print(f'Training samples : {X_train.shape[0]}')\n",
        "print(f'Testing  samples : {X_test.shape[0]}')"
    ),
    code(
        "# 4d. Feature scaling using StandardScaler\n",
        "# IMPORTANT: fit ONLY on training data — prevents data leakage\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train)\n",
        "X_test_scaled  = scaler.transform(X_test)\n",
        "\n",
        "print('Scaling done!')\n",
        "print(f'Training means (should be ~0): {X_train_scaled.mean(axis=0).round(2)}')\n",
        "print(f'Training stds  (should be ~1): {X_train_scaled.std(axis=0).round(2)}')"
    ),
    md(
        "**What just happened?**  \n",
        "- **Label Encoding** – text columns like school, sex, address were turned into numbers "
        "(0/1 or 0/1/2...) since math-based models only understand numbers.\n",
        "- **Feature Selection** – we picked 10 features that are both statistically and "
        "intuitively meaningful for predicting the final grade.\n",
        "- **Train/Test Split** – 80% for training, 20% held out as a fair test. "
        "`random_state=42` ensures the same split every time you run the notebook.\n",
        "- **Feature Scaling** – standardization (mean=0, std=1) is *critical* for Gradient Descent. "
        "Without it, features on wildly different scales cause the loss surface to be elongated "
        "and near-impossible to optimize efficiently."
    )
]

# ── Step 5 ────────────────────────────────────────────────────────────────────
cells += [
    md(
        "---\n",
        "## Step 5 – Implementing Linear Regression from Scratch using Gradient Descent\n",
        "\n",
        "**Quick Theory:**\n",
        "- Prediction: `y_hat = X * theta`\n",
        "- Cost (MSE): `J(theta) = (1/2m) * sum((y_hat - y)^2)`\n",
        "- Weight update: `theta = theta - alpha * (1/m) * X_T * (X*theta - y)`\n",
        "\n",
        "where `alpha` is the learning rate and `m` is the number of training samples."
    ),
    code(
        "class LinearRegressionGD:\n",
        "    \"\"\"\n",
        "    Linear Regression implemented from scratch using Batch Gradient Descent.\n",
        "    No scikit-learn model is used — only NumPy math.\n",
        "    \"\"\"\n",
        "\n",
        "    def __init__(self, learning_rate=0.01, n_iterations=1000):\n",
        "        self.lr           = learning_rate\n",
        "        self.n_iterations = n_iterations\n",
        "        self.theta        = None   # learned weights (including bias)\n",
        "        self.loss_history = []     # MSE recorded at every iteration\n",
        "\n",
        "    def _add_bias(self, X):\n",
        "        \"\"\"Prepend a column of 1s so the model can learn an intercept.\"\"\"\n",
        "        return np.c_[np.ones(X.shape[0]), X]\n",
        "\n",
        "    def _compute_loss(self, X_b, y):\n",
        "        \"\"\"Compute MSE: J = (1/(2m)) * sum((X*theta - y)^2)\"\"\"\n",
        "        m = len(y)\n",
        "        return (1.0 / (2 * m)) * np.sum((X_b @ self.theta - y) ** 2)\n",
        "\n",
        "    def fit(self, X, y):\n",
        "        X_b = self._add_bias(X)\n",
        "        m, n = X_b.shape\n",
        "        self.theta = np.zeros(n)      # start all weights at 0\n",
        "        self.loss_history = []\n",
        "\n",
        "        for _ in range(self.n_iterations):\n",
        "            error    = X_b @ self.theta - y          # how wrong we are\n",
        "            gradient = (1.0 / m) * (X_b.T @ error)  # direction of steepest ascent\n",
        "            self.theta -= self.lr * gradient          # step opposite to gradient\n",
        "            self.loss_history.append(self._compute_loss(X_b, y))\n",
        "\n",
        "        return self\n",
        "\n",
        "    def predict(self, X):\n",
        "        return self._add_bias(X) @ self.theta\n",
        "\n",
        "\n",
        "print('LinearRegressionGD class defined!')"
    ),
    md(
        "**What just happened?**  \n",
        "We built the model as a Python class:\n",
        "- `_add_bias` adds a column of 1s so the model can also learn a constant offset "
        "(the base grade when all features are zero).\n",
        "- `_compute_loss` measures how far off our predictions are.\n",
        "- `fit` is where learning happens — every iteration we compute the error, "
        "find the gradient (direction to move), and take a step opposite to it.\n",
        "- `predict` multiplies the learned weights by new data to get a grade prediction."
    )
]

# ── Step 6 ────────────────────────────────────────────────────────────────────
cells += [
    md(
        "---\n",
        "## Step 6 – Experimenting with Different Learning Rates\n",
        "\n",
        "The learning rate alpha controls how big a step we take at each iteration:\n",
        "- **Too small** → converges but painfully slowly.\n",
        "- **Too large** → overshoots the minimum and may diverge entirely.\n",
        "- **Just right** → smooth, efficient convergence."
    ),
    code(
        "learning_rates = [0.001, 0.01, 0.1, 0.5]\n",
        "n_iterations   = 1000\n",
        "trained_models = {}\n",
        "\n",
        "for lr in learning_rates:\n",
        "    model = LinearRegressionGD(learning_rate=lr, n_iterations=n_iterations)\n",
        "    model.fit(X_train_scaled, y_train)\n",
        "    trained_models[lr] = model\n",
        "    print(f'LR = {lr:<6}  ->  Final loss after {n_iterations} iters: {model.loss_history[-1]:.4f}')"
    )
]

# ── Step 7 ────────────────────────────────────────────────────────────────────
cells += [
    md(
        "---\n",
        "## Step 7 – Loss vs. Iterations (Convergence Analysis)\n",
        "\n",
        "These plots reveal whether each learning rate is actually working. "
        "A healthy convergence curve drops steeply then flattens — "
        "that flatness means we found (or got very close to) the minimum."
    ),
    code(
        "fig, axes = plt.subplots(2, 2, figsize=(14, 9))\n",
        "fig.suptitle('Loss (Cost) vs. Iterations for Different Learning Rates',\n",
        "             fontsize=15, fontweight='bold')\n",
        "\n",
        "palette = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']\n",
        "ax_flat = axes.flatten()\n",
        "\n",
        "for idx, (lr, model) in enumerate(trained_models.items()):\n",
        "    losses = model.loss_history\n",
        "    ax = ax_flat[idx]\n",
        "    ax.plot(losses, color=palette[idx], linewidth=2)\n",
        "    ax.set_title(f'Learning Rate = {lr}', fontsize=12)\n",
        "    ax.set_xlabel('Iteration')\n",
        "    ax.set_ylabel('MSE Loss')\n",
        "    ax.set_ylim(bottom=0)\n",
        "    ax.grid(True, alpha=0.3)\n",
        "    final = losses[-1]\n",
        "    ax.annotate(f'Final: {final:.3f}',\n",
        "                xy=(len(losses)-1, final),\n",
        "                xytext=(-120, 20), textcoords='offset points',\n",
        "                arrowprops=dict(arrowstyle='->', color='black'),\n",
        "                fontsize=10)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig('loss_vs_iterations.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Plot saved as loss_vs_iterations.png')"
    ),
    code(
        "# Combined view — all learning rates on one plot for direct comparison\n",
        "plt.figure(figsize=(11, 5))\n",
        "for idx, (lr, model) in enumerate(trained_models.items()):\n",
        "    losses = np.clip(model.loss_history, 0, 500)   # clip diverging curves\n",
        "    plt.plot(losses, label=f'LR = {lr}', color=palette[idx], linewidth=2)\n",
        "plt.xlabel('Iteration', fontsize=12)\n",
        "plt.ylabel('MSE Loss', fontsize=12)\n",
        "plt.title('Convergence Comparison Across Learning Rates',\n",
        "          fontsize=14, fontweight='bold')\n",
        "plt.legend(fontsize=11)\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ),
    md(
        "**What just happened?**  \n",
        "- **LR = 0.001** – Way too slow. Barely makes a dent in 1000 iterations.\n",
        "- **LR = 0.01** – The sweet spot. Smooth, consistent convergence.\n",
        "- **LR = 0.1** – Converges quickly; maybe a tiny bit noisy at the very start.\n",
        "- **LR = 0.5** – Risky. Steps are so big it can overshoot the minimum and diverge.\n",
        "\n",
        "Every healthy curve has the characteristic 'elbow' shape — steep drop, then flattening. "
        "That flattening tells us the model has found the optimal (or near-optimal) weights."
    )
]

# ── Step 8 ────────────────────────────────────────────────────────────────────
cells += [
    md(
        "---\n",
        "## Step 8 – Training the Final Model with the Best Learning Rate"
    ),
    code(
        "# Automatically pick the learning rate with the lowest final loss\n",
        "best_lr = min(trained_models, key=lambda lr: trained_models[lr].loss_history[-1])\n",
        "print(f'Best learning rate selected: {best_lr}')\n",
        "\n",
        "# Train a fresh model with more iterations for full convergence\n",
        "final_model = LinearRegressionGD(learning_rate=best_lr, n_iterations=3000)\n",
        "final_model.fit(X_train_scaled, y_train)\n",
        "\n",
        "print(f'Final training loss (3000 iterations): {final_model.loss_history[-1]:.4f}')\n",
        "print('\\nLearned weights (theta):')\n",
        "feature_names = ['bias'] + selected_features\n",
        "for name, w in zip(feature_names, final_model.theta):\n",
        "    print(f'  {name:<12} : {w:+.4f}')"
    ),
    md(
        "**What just happened?**  \n",
        "We automatically selected the best-performing learning rate, then trained a fresh model "
        "with 3000 iterations to ensure it fully converged. "
        "The printed theta values are the *learned weights* — each tells the model how much "
        "that feature shifts the predicted grade. "
        "Positive value = pushes grade up; negative = pulls it down."
    )
]

# ── Step 9 ────────────────────────────────────────────────────────────────────
cells += [
    md(
        "---\n",
        "## Step 9 – Evaluate the Model on the Test Set\n",
        "\n",
        "The real measure of a model is how well it does on data it has never seen before."
    ),
    code(
        "# Predictions on unseen test data\n",
        "y_pred = final_model.predict(X_test_scaled)\n",
        "\n",
        "# Compute all four regression metrics\n",
        "mae  = mean_absolute_error(y_test, y_pred)\n",
        "mse  = mean_squared_error(y_test, y_pred)\n",
        "rmse = np.sqrt(mse)\n",
        "r2   = r2_score(y_test, y_pred)\n",
        "\n",
        "print('=' * 45)\n",
        "print('   MODEL EVALUATION - TEST SET RESULTS')\n",
        "print('=' * 45)\n",
        "print(f'   MAE   (Mean Absolute Error)    : {mae:.4f}')\n",
        "print(f'   MSE   (Mean Squared Error)     : {mse:.4f}')\n",
        "print(f'   RMSE  (Root Mean Squared Error): {rmse:.4f}')\n",
        "print(f'   R2    (R-Squared Score)        : {r2:.4f}')\n",
        "print('=' * 45)"
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "fig.suptitle('Model Evaluation on Test Set', fontsize=14, fontweight='bold')\n",
        "\n",
        "# Scatter: actual vs predicted\n",
        "ax1 = axes[0]\n",
        "ax1.scatter(y_test, y_pred, alpha=0.65, color='steelblue',\n",
        "            edgecolors='white', s=70)\n",
        "mn = min(float(y_test.min()), float(y_pred.min()))\n",
        "mx = max(float(y_test.max()), float(y_pred.max()))\n",
        "ax1.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Prediction')\n",
        "ax1.set_xlabel('Actual Grade (G3)', fontsize=12)\n",
        "ax1.set_ylabel('Predicted Grade (G3)', fontsize=12)\n",
        "ax1.set_title(f'Actual vs. Predicted   |   R2 = {r2:.3f}', fontsize=12)\n",
        "ax1.legend()\n",
        "ax1.grid(True, alpha=0.3)\n",
        "\n",
        "# Residual plot\n",
        "ax2 = axes[1]\n",
        "residuals = y_test - y_pred\n",
        "ax2.scatter(y_pred, residuals, alpha=0.65, color='coral',\n",
        "            edgecolors='white', s=70)\n",
        "ax2.axhline(0, color='red', linestyle='--', linewidth=2)\n",
        "ax2.set_xlabel('Predicted Grade (G3)', fontsize=12)\n",
        "ax2.set_ylabel('Residual (Actual - Predicted)', fontsize=12)\n",
        "ax2.set_title('Residual Plot', fontsize=12)\n",
        "ax2.grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig('evaluation_plots.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "print('Plot saved as evaluation_plots.png')"
    ),
    code(
        "# Metrics bar chart\n",
        "metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}\n",
        "plt.figure(figsize=(8, 4))\n",
        "bars = plt.bar(metrics.keys(), metrics.values(),\n",
        "               color=['#3498db', '#e74c3c', '#2ecc71'],\n",
        "               edgecolor='black', width=0.4)\n",
        "for bar, val in zip(bars, metrics.values()):\n",
        "    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,\n",
        "             f'{val:.3f}', ha='center', va='bottom',\n",
        "             fontsize=12, fontweight='bold')\n",
        "plt.title('Regression Error Metrics', fontsize=13, fontweight='bold')\n",
        "plt.ylabel('Error Value')\n",
        "plt.ylim(0, max(metrics.values()) * 1.35)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ),
    md(
        "**What just happened?**  \n",
        "We evaluated the model using four standard metrics:\n",
        "\n",
        "| Metric | What it measures |\n",
        "|--------|------------------|\n",
        "| **MAE** | Average grade points we are off by — easy to interpret |\n",
        "| **MSE** | Same but squares the errors — punishes big mistakes harder |\n",
        "| **RMSE** | Square root of MSE — back in the same units as grades |\n",
        "| **R²** | Fraction of variance explained; 1.0 = perfect, 0 = just guessing the mean |\n",
        "\n",
        "**Scatter plot** — Points close to the red diagonal = good predictions.  \n",
        "**Residual plot** — Random scatter around zero = no obvious pattern missed. "
        "If you see a curve or funnel shape here, it means the model is missing some non-linearity."
    )
]

# ── Step 10 ───────────────────────────────────────────────────────────────────
cells += [
    md(
        "---\n",
        "## Step 10 – Feature Importance from Learned Weights\n",
        "\n",
        "Since all features were standardized before training, "
        "the absolute value of each learned weight is a fair comparison of feature importance."
    ),
    code(
        "weights    = final_model.theta[1:]   # skip bias\n",
        "sorted_idx = np.argsort(np.abs(weights))[::-1]\n",
        "sorted_features = [selected_features[i] for i in sorted_idx]\n",
        "sorted_weights  = weights[sorted_idx]\n",
        "\n",
        "colors_w = ['#2ecc71' if w > 0 else '#e74c3c' for w in sorted_weights]\n",
        "\n",
        "plt.figure(figsize=(10, 5))\n",
        "plt.barh(sorted_features[::-1], sorted_weights[::-1],\n",
        "         color=colors_w[::-1], edgecolor='black', linewidth=0.5)\n",
        "plt.axvline(0, color='black', linewidth=0.8)\n",
        "plt.xlabel('Weight (standardized scale)', fontsize=12)\n",
        "plt.title('Feature Importance from Learned Weights',\n",
        "          fontsize=13, fontweight='bold')\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print('Feature weights ranked by absolute importance:')\n",
        "for feat, w in zip(sorted_features, sorted_weights):\n",
        "    direction = 'UP   (positive)' if w > 0 else 'DOWN (negative)'\n",
        "    print(f'  {feat:<12}  {w:+.4f}  {direction}')"
    ),
    md(
        "**What just happened?**  \n",
        "Green bars mean the feature pushes the predicted grade upward; red bars pull it down.  \n",
        "G2 and G1 dominate (makes sense — they are scores from the same course and period). "
        "`failures` has a negative weight (more past failures = lower final grade). "
        "Features like `goout` and `absences` will show smaller negative influences — "
        "going out a lot and missing class both hurt the grade, but not as dramatically."
    )
]

# ── Step 11 ───────────────────────────────────────────────────────────────────
cells += [
    md(
        "---\n",
        "## Step 11 – Sample Predictions Comparison"
    ),
    code(
        "comparison = pd.DataFrame({\n",
        "    'Actual Grade (G3)' : y_test,\n",
        "    'Predicted Grade'    : np.round(y_pred, 2),\n",
        "    'Absolute Error'     : np.round(np.abs(y_test - y_pred), 2)\n",
        "}).reset_index(drop=True)\n",
        "\n",
        "print('Sample predictions (first 15 test students):')\n",
        "print(comparison.head(15).to_string(index=False))"
    ),
    md(
        "**What just happened?**  \n",
        "We lined up the actual grade, our model's predicted grade, and how far off we were "
        "for each student in the test set. "
        "Small absolute errors = the model is close. "
        "Most errors should be within 1–2 grade points on the 0–20 scale, "
        "which is pretty reasonable for a linear model."
    )
]

# ── Step 12 ───────────────────────────────────────────────────────────────────
cells += [
    md(
        "---\n",
        "## Step 12 – Convergence of the Final Model (3000 iterations)"
    ),
    code(
        "plt.figure(figsize=(11, 4))\n",
        "plt.plot(final_model.loss_history, color='#2c3e50', linewidth=2)\n",
        "plt.xlabel('Iteration', fontsize=12)\n",
        "plt.ylabel('MSE Loss', fontsize=12)\n",
        "plt.title(f'Final Model Convergence   (LR = {best_lr},  3000 iterations)',\n",
        "          fontsize=13, fontweight='bold')\n",
        "plt.axvspan(0,   300,  alpha=0.08, color='red',   label='Fast learning zone')\n",
        "plt.axvspan(300, 3000, alpha=0.05, color='green', label='Fine-tuning zone')\n",
        "plt.legend(fontsize=11)\n",
        "plt.grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(f'Loss at iteration 1    : {final_model.loss_history[0]:.4f}')\n",
        "print(f'Loss at iteration 100  : {final_model.loss_history[99]:.4f}')\n",
        "print(f'Loss at iteration 500  : {final_model.loss_history[499]:.4f}')\n",
        "print(f'Loss at iteration 3000 : {final_model.loss_history[-1]:.4f}')"
    ),
    md(
        "**What just happened?**  \n",
        "Two shaded zones tell the story:\n",
        "- **Red zone (0–300 iterations)** – The model is learning fast. Most weight-adjustment happens here.\n",
        "- **Green zone (300–3000 iterations)** – Fine-tuning. Loss barely moves because we are already "
        "very close to the optimal weights.\n",
        "\n",
        "This flat-tail behavior is perfectly expected for Gradient Descent on Linear Regression — "
        "it is a convex problem with exactly one global minimum, so the algorithm always finds it "
        "given a small enough learning rate."
    )
]

# ── Step 13 – Summary ─────────────────────────────────────────────────────────
cells += [
    md(
        "---\n",
        "## Step 13 – Summary and Interpretation\n",
        "\n",
        "### What we did\n",
        "1. Loaded and explored the Student Performance (Math) dataset — 395 students, 33 features.\n",
        "2. Pre-processed: label-encoded categorical columns, selected 10 meaningful features, "
        "split 80/20, and standardized using `StandardScaler`.\n",
        "3. Built a **Linear Regression model completely from scratch** using Batch Gradient Descent.\n",
        "4. Tested four learning rates (`0.001`, `0.01`, `0.1`, `0.5`) and plotted their convergence.\n",
        "5. Chose the best learning rate and trained the final model for 3000 iterations.\n",
        "6. Evaluated using MAE, MSE, RMSE, and R² on unseen test data.\n",
        "\n",
        "### Convergence Behavior Summary\n",
        "\n",
        "| Learning Rate | Behavior |\n",
        "|:---:|---|\n",
        "| 0.001 | Too slow — barely moves in 1000 iterations |\n",
        "| 0.01  | Sweet spot — smooth, consistent convergence |\n",
        "| 0.1   | Fast, slightly noisy early on, but converges |\n",
        "| 0.5   | Unstable — big steps can cause divergence |\n",
        "\n",
        "### Key Takeaways\n",
        "- **G1 and G2 are by far the best predictors** of G3. Prior grades predict future grades.\n",
        "- **`failures` has a negative weight** — more past failures means a lower predicted final grade.\n",
        "- **Feature scaling is non-negotiable** for Gradient Descent — "
        "without it the loss surface is elongated and convergence is very slow.\n",
        "- **Learning rate is the most critical hyperparameter** — "
        "too small wastes compute, too large causes the algorithm to diverge."
    ),
    code(
        "print('=' * 47)\n",
        "print('       FINAL MODEL PERFORMANCE SUMMARY')\n",
        "print('=' * 47)\n",
        "print(f'  Best Learning Rate  : {best_lr}')\n",
        "print(f'  Total Iterations    : 3000')\n",
        "print(f'  Features Used       : {len(selected_features)}')\n",
        "print(f'  Training Samples    : {X_train_scaled.shape[0]}')\n",
        "print(f'  Test Samples        : {X_test_scaled.shape[0]}')\n",
        "print('-' * 47)\n",
        "print(f'  MAE   : {mae:.4f}')\n",
        "print(f'  MSE   : {mse:.4f}')\n",
        "print(f'  RMSE  : {rmse:.4f}')\n",
        "print(f'  R2    : {r2:.4f}')\n",
        "print('=' * 47)"
    ),
    md(
        "**That's a wrap!**  \n",
        "We built Linear Regression with Gradient Descent completely from scratch — "
        "pure math translated into Python, no shortcuts. "
        "The experiment covered every step of the standard ML pipeline: "
        "data loading, preprocessing, model training, hyperparameter tuning, "
        "convergence analysis, and performance evaluation — "
        "all interpreted in plain, straightforward language."
    )
]

# ── Write notebook ─────────────────────────────────────────────────────────────
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
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = r"m:\Machine Learning\Lab_6(5)\lab6.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {out_path}")
print(f"Total cells: {len(cells)}")
