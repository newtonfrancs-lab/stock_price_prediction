# Stock Price Prediction Using TensorFlow

## Overview
This project uses Deep Learning and Long Short-Term Memory (LSTM) neural networks to predict stock closing prices based on historical stock market data.

The model analyses the previous 60 trading days to identify patterns and estimate the next closing price.

## Technologies Used
- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Model Architecture
The project uses a Sequential LSTM architecture consisting of:

- LSTM Layer (50 units)
- Dropout Layer (20%)
- LSTM Layer (50 units)
- Dropout Layer (20%)
- Dense Layer (25 neurons)
- Output Dense Layer (1 neuron)

## Methodology

1. Load historical stock market data from CSV.
2. Clean and preprocess the dataset.
3. Handle missing values and duplicate records.
4. Normalise closing prices using MinMaxScaler.
5. Create sequences using the previous 60 trading days.
6. Split the data into 80% training and 20% testing.
7. Train the LSTM model using TensorFlow.
8. Predict stock closing prices.
9. Evaluate the model using MAE, RMSE and R².

## Training Configuration

| Parameter | Value |
|---|---|
| Sequence Length | 60 days |
| Training Data | 80% |
| Testing Data | 20% |
| Epochs | 20 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss Function | Mean Squared Error |

## Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

## How to Run

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook or Python implementation to train and evaluate the model.

## Future Improvements
- Integrate real-time stock market data.
- Experiment with different LSTM architectures.
- Include additional indicators such as trading volume and moving averages.
- Develop a web-based stock prediction dashboard.

## Disclaimer
This project is developed for educational purposes. Predictions are based on historical data and should not be considered financial advice.
