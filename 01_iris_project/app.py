"""
app.py — Flask web server for the Iris Flower Classifier UI
Run with: ../.venv/bin/python app.py
Then open: http://localhost:5050
"""

import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# ── Train models once at startup ──────────────────────────────
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
dt_acc = accuracy_score(y_test, dt_model.predict(X_test))

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn_model.predict(X_test))

SPECIES = ['setosa', 'versicolor', 'virginica']
SPECIES_INFO = {
    'setosa': {
        'emoji': '🌸',
        'color': '#FF6B9D',
        'description': 'Small, compact flower. Has the smallest petals of all three species.',
        'fun_fact': 'Setosa is the easiest to identify — its tiny petals make it stand out!'
    },
    'versicolor': {
        'emoji': '🌺',
        'color': '#4ECDC4',
        'description': 'Medium-sized flower. Falls between Setosa and Virginica in measurements.',
        'fun_fact': 'Versicolor means "multicolored" in Latin — its petals often show multiple hues!'
    },
    'virginica': {
        'emoji': '💜',
        'color': '#7C5CBF',
        'description': 'The largest of the three. Has the biggest petals and sepals.',
        'fun_fact': 'Virginica was named after the U.S. state of Virginia where it was first catalogued!'
    }
}

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    dataset_preview = df.head(10).to_dict(orient='records')
    stats = {
        'total_samples': len(df),
        'features': list(iris.feature_names),
        'species': SPECIES,
        'dt_accuracy': round(dt_acc * 100, 1),
        'knn_accuracy': round(knn_acc * 100, 1),
        'species_counts': df['species'].value_counts().to_dict()
    }
    return render_template('index.html', stats=stats, species_info=SPECIES_INFO,
                           dataset=dataset_preview)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = pd.DataFrame([[
        float(data['sepal_length']),
        float(data['sepal_width']),
        float(data['petal_length']),
        float(data['petal_width'])
    ]], columns=iris.feature_names)

    # Decision Tree prediction
    dt_pred = int(dt_model.predict(features)[0])
    dt_proba = dt_model.predict_proba(features)[0].tolist()

    # KNN prediction
    knn_pred = int(knn_model.predict(features)[0])
    knn_proba = knn_model.predict_proba(features)[0].tolist()

    return jsonify({
        'dt': {
            'prediction': SPECIES[dt_pred],
            'probabilities': {s: round(p * 100, 1) for s, p in zip(SPECIES, dt_proba)},
            'accuracy': round(dt_acc * 100, 1)
        },
        'knn': {
            'prediction': SPECIES[knn_pred],
            'probabilities': {s: round(p * 100, 1) for s, p in zip(SPECIES, knn_proba)},
            'accuracy': round(knn_acc * 100, 1)
        },
        'species_info': SPECIES_INFO
    })


@app.route('/dataset')
def dataset():
    """Return full dataset as JSON for the UI table"""
    records = []
    for _, row in df.iterrows():
        records.append({
            'sepal_length': round(row['sepal length (cm)'], 1),
            'sepal_width': round(row['sepal width (cm)'], 1),
            'petal_length': round(row['petal length (cm)'], 1),
            'petal_width': round(row['petal width (cm)'], 1),
            'species': SPECIES[int(row['species'])]
        })
    return jsonify(records)


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  🌸  Iris Classifier UI is starting...")
    print("  Open your browser at: http://localhost:5050")
    print("="*50 + "\n")
    app.run(debug=True, port=5050)
