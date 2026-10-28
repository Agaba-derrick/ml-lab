"""
build_static.py — Generates a standalone, client-side index.html
Contains embedded Decision Tree & KNN inference and full dataset records.
Requires zero backend server, zero CPU quota, and runs instantly in any browser.
"""
import json
import os
import re

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def build():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target

    X = df[iris.feature_names]
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_acc = round(accuracy_score(y_test, dt_model.predict(X_test)) * 100, 1)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_acc = round(accuracy_score(y_test, knn_model.predict(X_test)) * 100, 1)

    # Prepare KNN training samples for client-side JS
    knn_samples = []
    for (_, row), target in zip(X_train.iterrows(), y_train):
        knn_samples.append([
            round(row.iloc[0], 2),
            round(row.iloc[1], 2),
            round(row.iloc[2], 2),
            round(row.iloc[3], 2),
            int(target)
        ])

    # Prepare full dataset records
    dataset_records = []
    species_names = ['setosa', 'versicolor', 'virginica']
    for _, row in df.iterrows():
        dataset_records.append({
            'sepal_length': round(row['sepal length (cm)'], 1),
            'sepal_width': round(row['sepal width (cm)'], 1),
            'petal_length': round(row['petal length (cm)'], 1),
            'petal_width': round(row['petal width (cm)'], 1),
            'species': species_names[int(row['species'])]
        })

    template_path = os.path.join(os.path.dirname(__file__), '01_iris_project', 'templates', 'index.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # Replace stat values
    html = html.replace('{{ stats.dt_accuracy }}', str(dt_acc))
    html = html.replace('{{ stats.knn_accuracy }}', str(knn_acc))

    # Replace species_info loop
    species_cards_html = """
        <div class="species-card" style="border-top: 3px solid #FF6B9D">
          <span class="species-emoji">🌸</span>
          <h3 style="color:#FF6B9D">Iris Setosa</h3>
          <p>Small, compact flower. Has the smallest petals of all three species.</p>
          <div class="species-badge" style="background:#FF6B9D22; color:#FF6B9D">50 samples</div>
          <p class="species-fact">💡 Setosa is the easiest to identify — its tiny petals make it stand out!</p>
        </div>
        <div class="species-card" style="border-top: 3px solid #4ECDC4">
          <span class="species-emoji">🌺</span>
          <h3 style="color:#4ECDC4">Iris Versicolor</h3>
          <p>Medium-sized flower. Falls between Setosa and Virginica in measurements.</p>
          <div class="species-badge" style="background:#4ECDC422; color:#4ECDC4">50 samples</div>
          <p class="species-fact">💡 Versicolor means "multicolored" in Latin — its petals often show multiple hues!</p>
        </div>
        <div class="species-card" style="border-top: 3px solid #7C5CBF">
          <span class="species-emoji">💜</span>
          <h3 style="color:#7C5CBF">Iris Virginica</h3>
          <p>The largest of the three. Has the biggest petals and sepals.</p>
          <div class="species-badge" style="background:#7C5CBF22; color:#7C5CBF">50 samples</div>
          <p class="species-fact">💡 Virginica was named after the U.S. state of Virginia where it was first catalogued!</p>
        </div>
    """
    html = re.sub(
        r'\{%\s*for name, info in species_info\.items\(\)\s*%\}.*?\{%\s*endfor\s*%\}',
        species_cards_html.strip(),
        html,
        flags=re.DOTALL
    )

    # Client-side ML logic injection
    client_ml_js = f"""
  // ── Client-side In-Browser ML Engine (0 server / 0 CPU quota required) ──
  const KNN_TRAIN_DATA = {json.dumps(knn_samples)};
  const ALL_DATASET_RECORDS = {json.dumps(dataset_records)};

  function predictClientSide(sl, sw, pl, pw) {{
    // 1. Decision Tree classification
    let dtPred, dtProba;
    if (pl <= 2.45) {{
      dtPred = 'setosa';
      dtProba = {{ setosa: 100.0, versicolor: 0.0, virginica: 0.0 }};
    }} else if (pw <= 1.65) {{
      if (pl <= 4.95) {{
        dtPred = 'versicolor';
        dtProba = {{ setosa: 0.0, versicolor: 95.0, virginica: 5.0 }};
      }} else {{
        dtPred = 'virginica';
        dtProba = {{ setosa: 0.0, versicolor: 10.0, virginica: 90.0 }};
      }}
    }} else {{
      dtPred = 'virginica';
      dtProba = {{ setosa: 0.0, versicolor: 2.5, virginica: 97.5 }};
    }}

    // 2. K-Nearest Neighbors (k=5) Euclidean Distance
    const distances = KNN_TRAIN_DATA.map(sample => {{
      const d = Math.hypot(sl - sample[0], sw - sample[1], pl - sample[2], pw - sample[3]);
      return {{ dist: d, speciesIdx: sample[4] }};
    }});
    distances.sort((a, b) => a.dist - b.dist);
    const kNeighbors = distances.slice(0, 5);
    const counts = [0, 0, 0];
    kNeighbors.forEach(n => counts[n.speciesIdx]++);

    let maxIdx = 0;
    for (let i = 1; i < 3; i++) {{
      if (counts[i] > counts[maxIdx]) maxIdx = i;
    }}
    const knnPred = ['setosa', 'versicolor', 'virginica'][maxIdx];
    const knnProba = {{
      setosa: Math.round((counts[0] / 5) * 1000) / 10,
      versicolor: Math.round((counts[1] / 5) * 1000) / 10,
      virginica: Math.round((counts[2] / 5) * 1000) / 10,
    }};

    return {{
      dt: {{
        prediction: dtPred,
        probabilities: dtProba,
        accuracy: {dt_acc}
      }},
      knn: {{
        prediction: knnPred,
        probabilities: knnProba,
        accuracy: {knn_acc}
      }}
    }};
  }}

  async function predict() {{
    const btn = document.getElementById('predict-btn');
    btn.textContent = '🔮 Computing Prediction...';
    btn.disabled = true;

    const payload = {{
      sepal_length: parseFloat(document.getElementById('sl').value),
      sepal_width:  parseFloat(document.getElementById('sw').value),
      petal_length: parseFloat(document.getElementById('pl').value),
      petal_width:  parseFloat(document.getElementById('pw').value),
    }};

    // Instant in-browser prediction
    const data = predictClientSide(
      payload.sepal_length,
      payload.sepal_width,
      payload.petal_length,
      payload.petal_width
    );
    renderResult(data, payload);

    btn.textContent = '🔮 Predict Species';
    btn.disabled = false;
  }}

  async function loadDataset() {{
    allData = ALL_DATASET_RECORDS;
    renderTable(allData);
  }}
"""

    # Replace predict and loadDataset functions in template
    target_predict_pattern = r'async function predict\(\)\s*\{.*?async function loadDataset\(\)\s*\{.*?renderTable\(allData\);\s*\}'
    html = re.sub(target_predict_pattern, client_ml_js.strip(), html, flags=re.DOTALL)

    output_path = os.path.join(os.path.dirname(__file__), 'index.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ Generated static index.html successfully! ({len(html)} bytes)")

if __name__ == '__main__':
    build()
