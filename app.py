from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)
CORS(app)  # Разрешаем CORS для всех маршрутов

@app.route('/clusters', methods=['POST'])
def cluster_data():
    data = request.get_json()
    locations = data.get('locations', [])
    num_clusters = data.get('num_clusters', 2)

    if len(locations) < num_clusters:
        return jsonify({'error': 'Недостаточно данных для кластеризации'}), 400

    locations = np.array(locations)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(locations)

    clusters = {
        'labels': kmeans.labels_.tolist(),
        'centers': kmeans.cluster_centers_.tolist(),
    }

    return jsonify(clusters)

if __name__ == '__main__':
    app.run(debug=True)
