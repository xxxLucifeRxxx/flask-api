from flask import Flask, request, jsonify
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)

@app.route('/api/clusterize', methods=['POST'])
def clusterize():
    data = request.json
    locations = np.array(data.get('locations', []))
    num_clusters = data.get('num_clusters', 2)

    if len(locations) < 2:
        return jsonify({'error': 'Недостаточно точек для кластеризации'}), 400

    # Выполняем кластеризацию
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(locations)
    centers = kmeans.cluster_centers_.tolist()
    labels = kmeans.labels_.tolist()

    # Формируем ответ
    clusters = {i: [] for i in range(num_clusters)}
    for idx, label in enumerate(labels):
        clusters[label].append(locations[idx].tolist())

    return jsonify({
        'centers': centers,
        'clusters': clusters
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
