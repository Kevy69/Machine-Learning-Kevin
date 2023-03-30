import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

model = joblib.load('Data/best_model.pkl')

test_samples = pd.read_csv('Data/test_samples.csv')
scaling_data = pd.read_csv('Data/for_scaling.csv')

# Fit_transform on scaling_data and than fit on test_samples
scaler = StandardScaler()
scaler.fit_transform(scaling_data)
scaled_test_samples = scaler.transform(test_samples)

# get probability and prediction
pred_prob = model.predict_proba(scaled_test_samples)
pred = model.predict(scaled_test_samples)


df = pd.DataFrame({
    'probability class 0': pred_prob[:, 0],
    'probability class 1': pred_prob[:, 1],
    'prediction': pred
})


df.to_csv(path_or_buf='Data/prediction.csv', index=False)