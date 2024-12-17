from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load the interpolated data
df = pd.read_csv("data/processed/4_resampled_cgm.csv")

# Normalize glucose values
scaler = MinMaxScaler(feature_range=(0, 1))  # Use z-score if preferred
df["glc"] = scaler.fit_transform(df["glc"].values.reshape(-1, 1))

# Save the scaler for inverse transformations later (optional)
import joblib
joblib.dump(scaler, "data/processed/scaler.pkl")

# After scaling, set missing values to -1
df["glc"] = df["glc"].fillna(-1)  

# Save normalized data
df.to_csv("data/processed/5_normalized_cgm.csv", index=False)
print("Data normalized and saved!")
