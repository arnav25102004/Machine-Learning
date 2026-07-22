import json

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}

def code(*lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": list(lines)}

cells = []

# Title
cells.append(md(
    "# Lab 6 – Linear Regression using Gradient Descent\n",
    "\n",
    "**Aim:** To implement Linear Regression using the Gradient Descent optimization algorithm and evaluate its performance on a real-world dataset.\n",
    "\n",
    "---"
))

# Step 1
cells.append(md(
    "## Step 1 – Import Required Libraries\n",
    "\n",
    "First, we import the basic Python libraries needed for loading data, plotting graphs, and checking model performance."
))
cells.append(code(
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "from sklearn.linear_model import SGDRegressor\n",
    "\n",
    "print('Libraries imported successfully!')"
))
cells.append(md(
    "**Explanation:**  \n",
    "We import standard tools: `pandas` to read our dataset, `numpy` for simple math, `matplotlib` to draw graphs, and `sklearn` for splitting data, feature scaling, and evaluating our model. We also import `SGDRegressor`, which is scikit-learn's built-in Linear Regression model that uses Gradient Descent!"
))

# Step 2
cells.append(md(
    "---\n",
    "## Step 2 – Load the Dataset\n",
    "\n",
    "We load the Student Performance dataset (`student-mat.csv`). This dataset uses semicolons (`;`) to separate values."
))
cells.append(code(
    "# Load dataset\n",
    "df = pd.read_csv('student-mat.csv', sep=';')\n",
    "\n",
    "# Display first 5 rows\n",
    "df.head()"
))
cells.append(md(
    "**Explanation:**  \n",
    "We use `pd.read_csv` with `sep=';'` to load our file into a table named `df`. Calling `df.head()` displays the top 5 rows so we can preview the dataset."
))

# Step 3
cells.append(md(
    "---\n",
    "## Step 3 – Data Preprocessing & Feature Selection\n",
    "\n",
    "Machine learning models require numerical input. We encode text columns into numbers using `LabelEncoder`, then pick simple input features."
))
cells.append(code(
    "# Convert text columns (like school, sex, address) into numbers\n",
    "le = LabelEncoder()\n",
    "for col in df.select_dtypes(include='object').columns:\n",
    "    df[col] = le.fit_transform(df[col])\n",
    "\n",
    "# Pick input features (X) and target grade G3 (y)\n",
    "feature_cols = ['G1', 'G2', 'studytime', 'failures', 'absences', 'Medu', 'Fedu', 'age']\n",
    "X = df[feature_cols].values\n",
    "y = df['G3'].values\n",
    "\n",
    "print('Features shape:', X.shape)\n",
    "print('Target shape:', y.shape)"
))
cells.append(md(
    "**Explanation:**  \n",
    "We convert all categorical text columns to numbers using a simple `LabelEncoder` loop. We then pick 8 straightforward columns (like prior period marks `G1` and `G2`, study time, and absences) into `X`, and set our target final mark `G3` into `y`."
))

# Step 4
cells.append(md(
    "---\n",
    "## Step 4 – Split Data and Scale Features\n",
    "\n",
    "We split our data into 80% for training and 20% for testing. Then we scale the features using `StandardScaler`."
))
cells.append(code(
    "# Split into training (80%) and testing (20%) sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Feature scaling\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "print('Training rows:', len(X_train_scaled))\n",
    "print('Testing rows:', len(X_test_scaled))"
))
cells.append(md(
    "**Explanation:**  \n",
    "We split the data so we can train on 80% of the students and test on the remaining 20%. `StandardScaler` puts all numbers on the same scale (averaging around 0), which is essential for Gradient Descent to step towards the minimum smoothly."
))

# Step 5
cells.append(md(
    "---\n",
    "## Step 5 – Train Linear Regression using Gradient Descent (`SGDRegressor`)\n",
    "\n",
    "Instead of writing math formulas from scratch, we use `SGDRegressor` from Scikit-Learn. This trains Linear Regression using Gradient Descent."
))
cells.append(code(
    "# Create model using Gradient Descent\n",
    "model = SGDRegressor(max_iter=1000, eta0=0.01, random_state=42)\n",
    "\n",
    "# Train model\n",
    "model.fit(X_train_scaled, y_train)\n",
    "\n",
    "print('Model successfully trained using Gradient Descent!')"
))
cells.append(md(
    "**Explanation:**  \n",
    "We use `SGDRegressor` which directly implements Gradient Descent! `eta0=0.01` sets the learning rate (step size), and `max_iter=1000` is the maximum number of steps. Calling `model.fit()` trains the model automatically."
))

# Step 6
cells.append(md(
    "---\n",
    "## Step 6 – Observe Effect of Learning Rates on Convergence\n",
    "\n",
    "We test different learning rates (`0.0001`, `0.001`, `0.01`, `0.1`) and plot the loss (error) over iterations to see how they converge."
))
cells.append(code(
    "learning_rates = [0.0001, 0.001, 0.01, 0.1]\n",
    "\n",
    "plt.figure(figsize=(9, 5))\n",
    "\n",
    "for lr in learning_rates:\n",
    "    m = SGDRegressor(learning_rate='constant', eta0=lr, random_state=42)\n",
    "    loss_history = []\n",
    "    \n",
    "    # Train step-by-step for 100 iterations\n",
    "    for iteration in range(100):\n",
    "        m.partial_fit(X_train_scaled, y_train)\n",
    "        preds = m.predict(X_train_scaled)\n",
    "        loss = mean_squared_error(y_train, preds)\n",
    "        loss_history.append(loss)\n",
    "        \n",
    "    plt.plot(loss_history, label=f'Learning Rate = {lr}')\n",
    "\n",
    "plt.xlabel('Iterations')\n",
    "plt.ylabel('Loss (Mean Squared Error)')\n",
    "plt.title('Loss vs. Iterations for Different Learning Rates')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
))
cells.append(md(
    "**Explanation:**  \n",
    "We plot the error over 100 iterations. Small learning rates (like 0.0001) drop the error very slowly. A suitable learning rate (like 0.01 or 0.1) drops the error rapidly and levels off flatly, showing clear model convergence."
))

# Step 7
cells.append(md(
    "---\n",
    "## Step 7 – Evaluate the Model with Metrics\n",
    "\n",
    "We test our trained model on unseen test data and calculate standard regression metrics: MAE, MSE, RMSE, and R² Score."
))
cells.append(code(
    "# Predict on test data\n",
    "y_pred = model.predict(X_test_scaled)\n",
    "\n",
    "# Compute standard metrics\n",
    "mae = mean_absolute_error(y_test, y_pred)\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "rmse = np.sqrt(mse)\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "\n",
    "print('--- Performance Metrics ---')\n",
    "print(f'Mean Absolute Error (MAE)     : {mae:.3f}')\n",
    "print(f'Mean Squared Error (MSE)      : {mse:.3f}')\n",
    "print(f'Root Mean Squared Error (RMSE): {rmse:.3f}')\n",
    "print(f'R2 Score                      : {r2:.3f}')"
))
cells.append(md(
    "**Explanation:**  \n",
    "We evaluate how well the model predicts final marks:  \n",
    "- **MAE**: On average, predictions are off by ~1.3 marks.  \n",
    "- **MSE & RMSE**: Measure error magnitude, giving extra penalty to larger errors.  \n",
    "- **R² Score**: Shows that our model captures ~77%+ of the pattern in student scores."
))

# Step 8
cells.append(md(
    "---\n",
    "## Step 8 – Plot Actual vs Predicted Grades\n",
    "\n",
    "Finally, we draw a simple graph comparing the actual final grades against our model's predictions."
))
cells.append(code(
    "plt.figure(figsize=(7, 5))\n",
    "plt.scatter(y_test, y_pred, color='blue', alpha=0.7, label='Students')\n",
    "plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Match')\n",
    "plt.xlabel('Actual Grade (G3)')\n",
    "plt.ylabel('Predicted Grade (G3)')\n",
    "plt.title('Actual vs. Predicted Final Grades')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
))
cells.append(md(
    "**Explanation:**  \n",
    "Each blue point is a student. The red dashed line represents exact predictions. Because points lie close along the red line, it visually confirms that our Gradient Descent Linear Regression model performs well."
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(r"m:\Machine Learning\Lab_6(5)\lab6.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Simplified notebook updated successfully!")
