from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model data
model_data = joblib.load('model_resep_final.pkl')
kamus = model_data['kamus_normalisasi']
df = model_data['dataframe']

# Fungsi Jaccard
def hitung_jaccard_similarity(set_user, set_resep):
    intersection = len(set_user & set_resep)
    union = len(set_user | set_resep)
    return intersection / union if union != 0 else 0

# Proses rekomendasi
def proses_model_rekomendasi(df, user_ingredients, kamus):
    user_normalized = [kamus.get(b, b) for b in user_ingredients]
    user_set = set(user_normalized)

    df = df.copy()
    df["Jaccard"] = df["Ingredients List"].apply(
        lambda resep: hitung_jaccard_similarity(user_set, set(resep))
    )

    hasil = df.sort_values(by=["Jaccard", "Loves"], ascending=False).reset_index(drop=True)
    return hasil[["Title", "Ingredients List", "Steps", "Jaccard", "Loves"]]

# Root => langsung render form
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint JSON (API)
@app.route('/rekomendasi', methods=['POST'])
def rekomendasi():
    data = request.get_json()
    if not data or 'ingredients' not in data:
        return jsonify({'error': 'Missing "ingredients" in request'}), 400

    user_ingredients = data['ingredients']
    try:
        hasil = proses_model_rekomendasi(df, user_ingredients, kamus)
        top5 = hasil.head(5)

        hasil_json = [
            {
                'title': row['Title'],
                'bahan': row['Ingredients List'],
                'jaccard': round(row['Jaccard'], 2),
                'loves': int(row['Loves'])
            }
            for _, row in top5.iterrows()
        ]

        return jsonify({'rekomendasi': hasil_json})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Proses hasil dari form
@app.route('/hasil', methods=['POST'])
def hasil():
    selected_ingredients = request.form.getlist('bahan')
    hasil = proses_model_rekomendasi(df, selected_ingredients, kamus)
    top5 = hasil.head(5)

    return render_template('index.html', rekomendasi=top5, input_user=selected_ingredients)

# Jalankan di Railway
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
