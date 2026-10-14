# 🌸 Project 1: Iris Flower Classifier

[![ML Lab CI Pipeline](https://github.com/Agaba-derrick/ml-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Agaba-derrick/ml-lab/actions/workflows/ci.yml)

> **Your first Machine Learning project!** Classify Iris flowers into 3 species using just 4 measurements — with a beautiful interactive web UI.

---

## 🚀 How to Run

### 1. Start the Web UI (Recommended)
```bash
# From the ml-lab/ directory:
.venv/bin/python 01_iris_project/app.py
```
Then open your browser at: **http://localhost:5050**

### 2. Run the Script (Terminal only)
```bash
.venv/bin/python 01_iris_project/iris_project.py
```
Charts will be saved to `01_iris_project/outputs/`

### 3. Run Automated Tests & Quality Checks (CI)
```bash
# Run all unit and integration tests
.venv/bin/pytest -v

# Run code style and linter checks
.venv/bin/ruff check .
```

---

## 📁 Project Structure

```
01_iris_project/
├── iris_project.py      ← Main ML script (all 7 steps explained)
├── app.py               ← Flask web server
├── templates/
│   └── index.html       ← Interactive UI (predictor, glossary, samples)
├── outputs/             ← Generated charts (created when you run the script)
│   ├── 01_feature_distributions.png
│   ├── 02_scatter_plots.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_confusion_matrices.png
│   └── 05_model_comparison.png
└── README.md            ← This file
```

---

## 🧠 What You'll Learn

| Concept | What It Means (Simply) |
|---------|------------------------|
| **Dataset** | A table of examples. Rows = flowers. Columns = measurements + species name |
| **Feature** | One measurement the model uses to predict (e.g. petal length) |
| **Label / Target** | What we're trying to predict — the species name |
| **Training** | Showing the model examples so it learns patterns |
| **Testing** | Checking the model on data it's never seen |
| **Accuracy** | How often the model gets it right (97% = 97 correct out of 100) |
| **Decision Tree** | A model that asks yes/no questions like a flowchart |
| **KNN** | Finds the 5 most similar known flowers and takes a majority vote |
| **Confusion Matrix** | A table showing which species the model got right vs. confused |

---

## 🌺 The Dataset

The **Iris dataset** is the "Hello World" of machine learning. It was collected by biologist Edgar Anderson in 1935 and has been used to teach ML ever since.

| Property | Value |
|----------|-------|
| Total samples | **150** (50 per species) |
| Features | **4** (sepal length, sepal width, petal length, petal width) |
| Classes | **3** (Setosa, Versicolor, Virginica) |
| Missing values | **None** (perfect clean data!) |

### The 4 Features Explained

```
🟢 Sepal Length (4.3 – 7.9 cm)
   The outer leaf-like parts of the flower.
   They protect the bud before it blooms.

🟩 Sepal Width (2.0 – 4.4 cm)
   How wide the sepal is at its widest point.
   Setosa tends to have relatively wide sepals.

🌸 Petal Length (1.0 – 6.9 cm)  ← MOST IMPORTANT!
   The colorful inner parts of the flower.
   Setosa = tiny (≈1.4cm), Virginica = huge (≈5.5cm)

💗 Petal Width (0.1 – 2.5 cm)
   How wide the petal is.
   Together with petal length, almost perfectly separates species.
```

### The 3 Species

| Species | Petal Length | Petal Width | Key Trait |
|---------|-------------|-------------|-----------|
| 🌸 **Setosa** | 1.0–1.9 cm | 0.1–0.6 cm | Tiny petals — easiest to identify |
| 🌺 **Versicolor** | 3.0–5.1 cm | 1.0–1.8 cm | Medium — sits in the middle |
| 💜 **Virginica** | 4.5–6.9 cm | 1.4–2.5 cm | Largest petals — biggest flower |

---

## 🤖 The Two Models

### 🌳 Decision Tree
Works like a flowchart of yes/no questions:
```
Is petal length < 2.45 cm?
├── YES → 🌸 Setosa (100% sure!)
└── NO  → Is petal width < 1.75 cm?
          ├── YES → 🌺 Versicolor
          └── NO  → 💜 Virginica
```
**Accuracy on test data: 96.7%** (got 29/30 right)

### 🔵 K-Nearest Neighbors (KNN, k=5)
Finds the 5 most similar flowers from training data, majority vote wins:
```
New flower measurements → Find 5 closest training samples
4 of them are Versicolor, 1 is Virginica → Predict: Versicolor
```
**Accuracy on test data: 100%** (got 30/30 right!)

---

## 🧪 Test Data — Try These Yourself!

Copy and paste these values into the web UI predictor sliders:

### 🌸 Setosa Examples (tiny petals)
```
Sample 1: Sepal=5.1×3.5, Petal=1.4×0.2  → Setosa (100% confident)
Sample 2: Sepal=4.6×3.1, Petal=1.5×0.2  → Setosa
Sample 3: Sepal=5.4×3.9, Petal=1.7×0.4  → Setosa
Sample 4: Sepal=4.9×3.1, Petal=1.5×0.1  → Setosa
```

### 🌺 Versicolor Examples (medium measurements)
```
Sample 1: Sepal=7.0×3.2, Petal=4.7×1.4  → Versicolor
Sample 2: Sepal=5.5×2.4, Petal=3.8×1.1  → Versicolor
Sample 3: Sepal=6.4×2.9, Petal=4.3×1.3  → Versicolor
Sample 4: Sepal=6.0×2.7, Petal=5.1×1.6  → Tricky! (borderline)
```

### 💜 Virginica Examples (large petals)
```
Sample 1: Sepal=6.3×3.3, Petal=6.0×2.5  → Virginica (100% confident)
Sample 2: Sepal=5.8×2.7, Petal=5.1×1.9  → Virginica
Sample 3: Sepal=7.1×3.0, Petal=5.9×2.1  → Virginica
Sample 4: Sepal=6.5×3.0, Petal=5.2×2.0  → Virginica
```

### 🎮 Experiments to Try
1. **Find the boundary:** Start at petal length 1.4 → increase slowly → at what value does Setosa become Versicolor?
2. **Model disagreement:** Try `Sepal=6.0×2.7, Petal=5.1×1.6` — do both models agree?
3. **Most important feature:** Hold everything constant at mid-range, change one feature at a time. Which one matters most?
4. **Impossible flower:** Try extreme values like petal=1.0 but sepal=7.9. What happens?

---

## 📊 Reading the Results (Confusion Matrix)

The confusion matrix from running the script shows:

```
                Predicted:
                Setosa  Versicolor  Virginica
Actual: Setosa    10        0           0       ← 10/10 correct ✅
     Versicolor    0        9           1       ← 1 confused with Virginica
      Virginica    0        0          10       ← 10/10 correct ✅
```

**Reading it:** Diagonal = correct. Off-diagonal = mistakes.
The Decision Tree confused 1 Versicolor sample as Virginica.

---

## 📈 What's Next on Your ML Journey?

```
✅ Project 1: Iris Classification   ← You are here
🔜 Project 2: House Price Regression (predict a NUMBER instead of a category)
🔜 Project 3: Titanic Survival (messy real-world data, feature engineering)
🔜 Project 4: Digit Recognition (your first image ML!)
🔜 Project 5: Neural Networks (deep learning begins!)
```

---

## 📦 Requirements

Install with:
```bash
.venv/bin/pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| `scikit-learn` | ML models (DecisionTree, KNN) and dataset |
| `pandas` | Data tables (DataFrames) |
| `matplotlib` | Charts and plots |
| `seaborn` | Beautiful statistical charts |
| `flask` | Web server for the UI |

---

*Dataset originally from UCI ML Repository. First used by R.A. Fisher in 1936.*
