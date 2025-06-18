import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Model ve veri dosyalarının yollarını kontrol edin
rf_model = joblib.load('XGBoostModel.pkl')
df = pd.read_parquet('../PreprocessedData/dataLog.parquet')

brands = df['Brand'].unique()
models = df['Model'].unique()
fuels = df['Fuel'].unique()

st.title('Araç Fiyatı Tahmin Uygulaması')

brand = st.selectbox('Marka', brands)
model = st.selectbox('Model', models)
power = st.number_input('Motor Gücü (Power)', value=0.0)
mileage = st.number_input('Kilometre (Mileage)', value=0.0)
fuel = st.selectbox('Yakıt Türü', fuels)

# Kodlanmış özellikleri alma
brand_encoded = df[df['Brand'] == brand]['Brand_encoded'].values[0]
model_encoded = df[df['Model'] == model]['Model_encoded'].values[0]
fuel_encoded = df[df['Fuel'] == fuel]['Fuel_encoded'].values[0]
transmission_manual = 1 if st.radio('Vites Türü (Manual)', ['Evet', 'Hayır']) == 'Evet' else 0

# Özellikleri hazırlama (Vehicle_Age kaldırıldı)
power_log = np.log(power) if power > 0 else 0
mileage_log = np.log(mileage) if mileage > 0 else 0

input_data = np.array([[power_log, mileage_log, brand_encoded, model_encoded, fuel_encoded, transmission_manual]])

# Tahmini hesaplama
predicted_log_price = rf_model.predict(input_data)
predicted_price = np.exp(predicted_log_price[0])

# Sonucu gösterme
st.write(f'Tahmin Edilen Fiyat: {predicted_price:.2f} USD')
