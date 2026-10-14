"""
=============================================================
  🌸 PROJECT 1: The Iris Flower Classifier
  Your First Machine Learning Project!
=============================================================
  What you'll learn in this project:
    1. How to load a dataset
    2. How to explore data (EDA - Exploratory Data Analysis)
    3. How to visualize data with charts
    4. How to train a Machine Learning model
    5. How to evaluate how good your model is
    6. How to make predictions on new data

  The Iris dataset:
    - 150 flower samples
    - 3 species: Setosa, Versicolor, Virginica
    - 4 features: sepal length, sepal width, petal length, petal width
    - Goal: given measurements, predict the flower species
=============================================================
"""

# ─────────────────────────────────────────────────────────────
# STEP 0: Import the tools we need
# ─────────────────────────────────────────────────────────────
# Think of imports like gathering your tools before starting work

import os
import warnings

import matplotlib.pyplot as plt  # For drawing charts
import pandas as pd  # For working with data tables
import seaborn as sns  # For prettier charts
from sklearn.datasets import load_iris  # The Iris dataset (built into scikit-learn)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split  # To split data into train/test
from sklearn.neighbors import KNeighborsClassifier  # Our second ML model
from sklearn.tree import DecisionTreeClassifier  # Our first ML model

warnings.filterwarnings('ignore')

# Make sure output folder exists
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  🌸  IRIS FLOWER CLASSIFIER — ML Learning Journey")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# STEP 1: Load the Dataset
# ─────────────────────────────────────────────────────────────
print("\n📦 STEP 1: Loading the Iris Dataset...")

# Load the dataset from scikit-learn
iris = load_iris()

# Convert it to a pandas DataFrame (a table) for easier handling
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names  # column names: sepal length, etc.
)

# Add the target column (the species label — what we want to predict)
df['species'] = iris.target

# Map numbers to actual species names (0 → setosa, 1 → versicolor, 2 → virginica)
df['species_name'] = df['species'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})

print(f"✅ Dataset loaded! Shape: {df.shape[0]} rows × {df.shape[1]} columns")


# ─────────────────────────────────────────────────────────────
# STEP 2: Explore the Data (EDA)
# ─────────────────────────────────────────────────────────────
print("\n🔍 STEP 2: Exploring the Data (EDA)...")

print("\n--- First 5 rows of the dataset ---")
print(df.head().to_string())

print("\n--- Basic statistics ---")
print(df.describe().to_string())

print("\n--- Count of each species (perfectly balanced!) ---")
print(df['species_name'].value_counts().to_string())

print("\n--- Are there any missing values? ---")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ No missing values — great, clean dataset!")
else:
    print(missing)


# ─────────────────────────────────────────────────────────────
# STEP 3: Visualize the Data
# ─────────────────────────────────────────────────────────────
print("\n📊 STEP 3: Creating Visualizations...")

# Set a beautiful style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'setosa': '#FF6B9D', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}

# --- Chart 1: Feature Distribution (Histogram) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('🌸 Iris Dataset — Feature Distributions by Species',
             fontsize=16, fontweight='bold', y=1.02)

features = ['sepal length (cm)', 'sepal width (cm)',
            'petal length (cm)', 'petal width (cm)']

for ax, feature in zip(axes.flatten(), features):
    for species_name, color in colors.items():
        subset = df[df['species_name'] == species_name][feature]
        ax.hist(subset, bins=15, alpha=0.6, color=color, label=species_name)
    ax.set_title(feature.replace(' (cm)', '').title(), fontweight='bold')
    ax.set_xlabel('cm')
    ax.set_ylabel('Count')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: outputs/01_feature_distributions.png")

# --- Chart 2: Scatter Plot (Petal features are most separable) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('🌸 Iris — Scatter Plots (Can we see patterns?)',
             fontsize=14, fontweight='bold')

for species_name, color in colors.items():
    subset = df[df['species_name'] == species_name]
    # Sepal features
    ax1.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'],
                color=color, label=species_name, alpha=0.7, s=60, edgecolors='white')
    # Petal features
    ax2.scatter(subset['petal length (cm)'], subset['petal width (cm)'],
                color=color, label=species_name, alpha=0.7, s=60, edgecolors='white')

ax1.set_xlabel('Sepal Length (cm)'); ax1.set_ylabel('Sepal Width (cm)')
ax1.set_title('Sepal Features (harder to separate)'); ax1.legend()

ax2.set_xlabel('Petal Length (cm)'); ax2.set_ylabel('Petal Width (cm)')
ax2.set_title('Petal Features (easier to separate!)'); ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_scatter_plots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: outputs/02_scatter_plots.png")

# --- Chart 3: Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap\n(1.0 = perfect positive, -1.0 = perfect negative)',
             fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: outputs/03_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────
# STEP 4: Prepare Data for Training
# ─────────────────────────────────────────────────────────────
print("\n⚙️  STEP 4: Preparing Data for Training...")

# X = features (the measurements) — what the model LEARNS FROM
# y = target  (the species)       — what the model TRIES TO PREDICT
X = df[features]
y = df['species']

# Split: 80% for training, 20% for testing
# random_state=42 means we get the same split every time (reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training set: {len(X_train)} samples")
print(f"  Testing set:  {len(X_test)} samples")
print("  (80% train / 20% test split)")


# ─────────────────────────────────────────────────────────────
# STEP 5: Train Machine Learning Models
# ─────────────────────────────────────────────────────────────
print("\n🤖 STEP 5: Training Machine Learning Models...")

# --- Model 1: Decision Tree ---
# Like a flowchart: "Is petal length > 2.5? → Yes → Is petal width > 1.8? → ..."
print("\n  Model 1: Decision Tree Classifier")
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)               # TRAINING happens here!
dt_predictions = dt_model.predict(X_test)    # Make predictions on test set
dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f"  ✅ Decision Tree Accuracy: {dt_accuracy * 100:.1f}%")

# --- Model 2: K-Nearest Neighbors (KNN) ---
# Finds the 5 most similar flowers it has seen and takes a majority vote
print("\n  Model 2: K-Nearest Neighbors (k=5)")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f"  ✅ KNN Accuracy: {knn_accuracy * 100:.1f}%")


# ─────────────────────────────────────────────────────────────
# STEP 6: Evaluate the Models
# ─────────────────────────────────────────────────────────────
print("\n📈 STEP 6: Evaluating Model Performance...")

species_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']

# --- Confusion Matrix for both models ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Confusion Matrices — How Often Was Each Model Correct?',
             fontsize=14, fontweight='bold')

for ax, predictions, model_name in zip(
    axes,
    [dt_predictions, knn_predictions],
    ['Decision Tree', 'KNN (k=5)']
):
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=species_names, yticklabels=species_names,
                linewidths=0.5, cbar=False)
    ax.set_xlabel('Predicted Species', fontweight='bold')
    ax.set_ylabel('Actual Species', fontweight='bold')
    accuracy = accuracy_score(y_test, predictions)
    ax.set_title(f'{model_name}\nAccuracy: {accuracy * 100:.1f}%', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: outputs/04_confusion_matrices.png")

# --- Detailed Classification Report ---
print("\n--- Decision Tree — Detailed Report ---")
print(classification_report(y_test, dt_predictions, target_names=species_names))

print("--- KNN — Detailed Report ---")
print(classification_report(y_test, knn_predictions, target_names=species_names))

# --- Model Comparison Chart ---
fig, ax = plt.subplots(figsize=(8, 5))
models = ['Decision Tree', 'KNN (k=5)']
accuracies = [dt_accuracy * 100, knn_accuracy * 100]
bar_colors = ['#FF6B9D', '#4ECDC4']

bars = ax.bar(models, accuracies, color=bar_colors, width=0.4,
              edgecolor='white', linewidth=1.5)
ax.set_ylim(0, 115)
ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_title('Model Comparison', fontsize=14, fontweight='bold')

# Add accuracy labels on bars
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)

ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3, label='Perfect (100%)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved: outputs/05_model_comparison.png")


# ─────────────────────────────────────────────────────────────
# STEP 7: Make a Real Prediction!
# ─────────────────────────────────────────────────────────────
print("\n🔮 STEP 7: Making Real Predictions with the Best Model!")

# Let's say we measured a new flower:
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # [sepal_length, sepal_width, petal_length, petal_width]

# Pick the best model (highest accuracy)
best_model = dt_model if dt_accuracy >= knn_accuracy else knn_model
best_name = "Decision Tree" if dt_accuracy >= knn_accuracy else "KNN"

prediction = best_model.predict(new_flower)
prediction_proba = best_model.predict_proba(new_flower)[0]

predicted_species = species_names[prediction[0]]
print(f"\n  Flower measurements: sepal={new_flower[0][0]}cm x {new_flower[0][1]}cm, "
      f"petal={new_flower[0][2]}cm x {new_flower[0][3]}cm")
print(f"  Predicted species: {predicted_species.upper()} (using {best_name})")
print("  Confidence scores:")
for name, prob in zip(species_names, prediction_proba):
    bar = 'X' * int(prob * 20)
    print(f"    {name:12s}: {bar:<20s} {prob*100:.1f}%")


# ─────────────────────────────────────────────────────────────
# DONE! Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PROJECT 1 COMPLETE! Here's what you learned:")
print("=" * 60)
print("  - Loading data with pandas and sklearn")
print("  - Exploring data (shape, stats, missing values)")
print("  - Visualizing data with matplotlib and seaborn")
print("  - Splitting data into training and testing sets")
print("  - Training two models: Decision Tree & KNN")
print("  - Evaluating models with accuracy & confusion matrix")
print("  - Making predictions on new, unseen data")
print()
print(f"  Decision Tree Accuracy: {dt_accuracy * 100:.1f}%")
print(f"  KNN Accuracy:           {knn_accuracy * 100:.1f}%")
print()
print("  Check the 'outputs/' folder for all the charts!")
print("=" * 60)
print("\nNext up: Regression — predicting numbers instead of categories!")
