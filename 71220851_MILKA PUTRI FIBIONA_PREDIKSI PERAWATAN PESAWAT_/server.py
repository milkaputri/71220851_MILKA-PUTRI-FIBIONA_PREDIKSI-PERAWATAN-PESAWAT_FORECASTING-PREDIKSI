import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def process_data(file_path):
    data = pd.read_csv(file_path)
    
    def check_data_quality(df):
        missing_values = df.isnull().sum()
        duplicate_values = df.duplicated().sum()
        outlier_values = {}
        for column in df.select_dtypes(include=[np.number]).columns:
            outliers = df[(df[column] < df[column].quantile(0.01)) | (df[column] > df[column].quantile(0.99))]
            outlier_values[column] = len(outliers)
        return missing_values, duplicate_values, outlier_values

    missing_values, duplicate_values, outlier_values = check_data_quality(data)
    
    def clean_data(df):
        df = df.fillna(df.median())
        df = df.drop_duplicates()
        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
            df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        return df

    cleaned_data = clean_data(data)
    
    def normalize_data(df):
        columns_to_normalize = df.drop(columns=['id', 'cycle', 'label_bnc', 'label_mcc', 'ttf']).columns
        scaler = MinMaxScaler()
        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
        return df

    normalized_data = normalize_data(cleaned_data)
    normalized_data = normalized_data.drop(columns=['id', 'cycle', 'label_bnc', 'label_mcc'])
    X = normalized_data.drop(columns=['ttf'])
    y = normalized_data['ttf']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # Menghitung persentase kerusakan yang diprediksi
    def predict_damage_percentage(features, model, max_ttf=200):
        ttf_pred = model.predict(features)
        ttf_pred = np.clip(ttf_pred, 0, max_ttf)  # Memastikan nilai yang diprediksi tidak melebihi max_ttf
        damage_percentage = 100 * (max_ttf - ttf_pred) / max_ttf
        return damage_percentage

    # Menghitung rata-rata persentase kerusakan yang diprediksi untuk set data uji
    avg_damage_percentage = np.mean(predict_damage_percentage(X_test, model))

    # Membuat histogram TTF yang diprediksi dan aktual
    plt.figure(figsize=(10, 6))
    plt.hist(y_test, bins=30, alpha=0.5, color='blue', label='TTF Aktual')
    plt.hist(y_pred_test, bins=30, alpha=0.5, color='red', label='TTF Diprediksi')
    plt.title('Distribusi Persentase Kerusakan Pesawat')
    plt.xlabel('Persentase Kerusakan')
    plt.ylabel('Jumlah')
    plt.legend()

    # Menyimpan plot ke buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return {
        "test_mse": test_mse,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "avg_damage_percentage": avg_damage_percentage,
        "plot": plot_data
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"result": "Tidak ada bagian file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"result": "Tidak ada file yang dipilih"}), 400
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        result = process_data(file_path)
        return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
