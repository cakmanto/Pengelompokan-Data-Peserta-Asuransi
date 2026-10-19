from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # 1. Read dataset
        df = pd.read_excel(filepath)
        df = df.dropna(subset=["Usia", "Pekerjaan", "Gaji"])

        # 2. Preprocessing
        scaler = StandardScaler()
        df["Usia"] = scaler.fit_transform(df[["Usia"]])
        df["Gaji"] = scaler.fit_transform(df[["Gaji"]])

        le = LabelEncoder()
        df["Pekerjaan"] = le.fit_transform(df["Pekerjaan"])

        # 3. Clustering DBSCAN
        X = df[["Usia", "Pekerjaan", "Gaji"]]
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(X)
        df["Cluster"] = clusters

        # 4. Evaluasi
        mask = clusters != -1
        if len(set(clusters[mask])) >= 2:
            sil = silhouette_score(X[mask], clusters[mask])
            dbi = davies_bouldin_score(X[mask], clusters[mask])
        else:
            sil, dbi = None, None

        # 5. Visualisasi
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=df["Usia"], y=df["Gaji"], hue=df["Cluster"], palette="tab10")
        plt.title("DBSCAN Clustering Usia vs Gaji")
        plt.savefig("static/cluster.png")

        return render_template("index.html", tables=df.head().to_html(), sil=sil, dbi=dbi, img="static/cluster.png")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)