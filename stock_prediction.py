
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. LOAD DATASET

data = pd.read_csv("GME_stock.csv")

print("Original columns:")
print(data.columns.tolist())

print("\nOriginal dataset shape:", data.shape)

# 2. CLEAN AND PREPROCESS DATA

# Rename columns to standard names
data = data.rename(columns={
    "date": "Date",
    "open_price": "Open",
    "high_price": "High",
    "low_price": "Low",
    "close_price": "Close",
    "volume": "Volume",
    "adjclose_price": "Adj Close"
})

# Convert date column
data["Date"] = pd.to_datetime(
    data["Date"],
    errors="coerce"
)

# Convert closing price to numeric
data["Close"] = pd.to_numeric(
    data["Close"],
    errors="coerce"
)

# Remove missing dates and closing prices
data = data.dropna(subset=["Date", "Close"])

# Remove invalid closing prices
data = data[data["Close"] > 0]

# Remove duplicate records
data = data.drop_duplicates()

# Sort from oldest date to newest date
data = data.sort_values("Date").reset_index(drop=True)

print("\nCleaned dataset:")
print(data[["Date", "Close"]].head())

print("\nTotal valid records:", len(data))

# 3. SELECT CLOSING PRICES

prices = data[["Close"]].values

sequence_length = 60

if len(prices) <= sequence_length:
    raise ValueError(
        f"At least {sequence_length + 1} valid records "
        "are required."
    )

# Chronological 80% training-period boundary
split_index = int(len(prices) * 0.8)

if split_index <= sequence_length:
    raise ValueError(
        "Not enough records in the training period."
    )

# 4. NORMALIZE DATA

# Fit the scaler only on the training period
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(prices[:split_index])

# Apply the fitted scaler to all prices
scaled_data = scaler.transform(prices)

# 5. CREATE SEQUENCES

X = []
y = []
target_indices = []

for i in range(sequence_length, len(scaled_data)):

    # Previous 60 closing prices
    X.append(
        scaled_data[i - sequence_length:i, 0]
    )

    # Closing price to predict
    y.append(scaled_data[i, 0])

    target_indices.append(i)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
target_indices = np.array(target_indices)

# Reshape for LSTM:
# samples, timesteps, features
X = X.reshape((X.shape[0], sequence_length, 1))

# 6. SPLIT TRAINING AND TESTING DATA

# Keep the split in chronological order
train_mask = target_indices < split_index
test_mask = target_indices >= split_index

X_train_all = X[train_mask]
y_train_all = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

# Use the final 10% of training sequences
# for validation during model training
validation_size = max(
    1, int(len(X_train_all) * 0.1)
)

if len(X_train_all) <= validation_size:
    raise ValueError(
        "Not enough training sequences for validation."
    )

X_train = X_train_all[:-validation_size]
y_train = y_train_all[:-validation_size]

X_val = X_train_all[-validation_size:]
y_val = y_train_all[-validation_size:]

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

# 7. BUILD LSTM MODEL

model = tf.keras.Sequential([
    tf.keras.Input(shape=(sequence_length, 1)),

    tf.keras.layers.LSTM(
        50,
        return_sequences=True
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(50),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(25),

    tf.keras.layers.Dense(1)
])

# 8. COMPILE MODEL

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

model.summary()

# 9. TRAIN MODEL

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    shuffle=False,
    verbose=1
)

# 10. PREDICT STOCK PRICES

predicted = model.predict(
    X_test,
    verbose=0
)

# Convert normalized prices to actual prices
predicted_prices = scaler.inverse_transform(predicted)

actual_prices = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

# 11. EVALUATE MODEL

mae = mean_absolute_error(
    actual_prices,
    predicted_prices
)

rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        predicted_prices
    )
)

r2 = r2_score(
    actual_prices,
    predicted_prices
)

print("\n========== MODEL RESULTS ==========")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"R2 Score : {r2:.4f}")

# 12. DISPLAY PREDICTIONS

test_dates = data["Date"].iloc[
    target_indices[test_mask]
].values

results = pd.DataFrame({
    "Date": test_dates,
    "Actual Closing Price": actual_prices.flatten(),
    "Predicted Closing Price": predicted_prices.flatten()
})

print("\nActual vs Predicted Closing Prices:")
print(results.head(10).to_string(index=False))

# 13. SAVE RESULTS

results.to_csv(
    "stock_prediction_results.csv",
    index=False
)

print("\nPredictions saved to stock_prediction_results.csv")

# Save the trained model
model.save("gme_lstm_model.keras")

print("Trained model saved to gme_lstm_model.keras")