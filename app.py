from flask import Flask, render_template, request
import joblib
import numpy as np

# Load models
rf_model = joblib.load('models/rf_model.pkl')
c45_model = joblib.load('models/c45_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/', methods=['POST'])
def predict():
    # Ambil data dari form input
    club_name = request.form['club_name']
    pts_g = float(request.form['pts_g'])
    xg = float(request.form['xg'])
    xga = float(request.form['xga'])
    xgd = float(request.form['xgd'])
    xgd_90 = float(request.form['xgd_90'])
    w = float(request.form['w'])

    # Buat data untuk prediksi
    input_data = np.array([[pts_g, xg, xga, xgd, xgd_90, w]])

    # Prediksi menggunakan setiap model
    rf_pred = rf_model.predict(input_data)[0]
    c45_pred = c45_model.predict(input_data)[0]
    xgb_pred = xgb_model.predict(input_data)[0]

    # Tentukan status untuk setiap model
    rf_status = "Layak masuk UCL" if rf_pred == 1 else "Tidak layak masuk UCL"
    c45_status = "Layak masuk UCL" if c45_pred == 1 else "Tidak layak masuk UCL"
    xgb_status = "Layak masuk UCL" if xgb_pred == 1 else "Tidak layak masuk UCL"

    # Kirim data ke halaman result.html
    return render_template('result.html', 
                           club_name=club_name, 
                           rf_status=rf_status, 
                           c45_status=c45_status, 
                           xgb_status=xgb_status)

if __name__ == "__main__":
    app.run(debug=True)
