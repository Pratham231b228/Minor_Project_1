# train_model.py
# (Run this file ONCE to create your model)

import yfinance as yf
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
import joblib
import numpy as np

print(" Starting model training...")

# --- Feature Engineering Function (Copied from your app) ---
def add_technical_indicators(df):
    df = df.copy()
    
    # 1. Moving Averages
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    
    # 2. Relative Strength Index (RSI)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # 3. Trend & Volatility
    df["Trend"] = df["Close"].shift(1) < df["Close"] 
    df["Volatility"] = df["Close"].pct_change().rolling(10).std()
    
    # 4. MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    return df

# --- 1. Fetch Training Data ---
# We train on a broad market ETF (S&P 500) to get a general-purpose model
# We use 5 years of data for a robust model.
print("Fetching 5 years of training data for 'SPY'...")
ticker = "SPY"
data = yf.Ticker(ticker).history(period="5y")
data = data.reset_index()
data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)

# --- 2. Create Features and Target ---
print("Adding technical indicators...")
df = add_technical_indicators(data)

# Target: Any positive gain in the next 5 days
df["Target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)

# Remove all rows with NaN (from rolling windows and target shift)
df = df.dropna()

# --- 3. Define Features and Split Data ---
features = ["SMA_10", "SMA_50", "RSI", "Volatility", "MACD", "Open", "High", "Low", "Close", "Volume"]
X = df[features]
y = df["Target"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training model on {len(X_train)} samples...")

# --- 4. Train the XGBoost Model ---
# (Using the same parameters as your app)
model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.05, 
    max_depth=5, 
    random_state=1, 
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# --- 5. Test and Save Model ---
preds = model.predict(X_test)
precision = precision_score(y_test, preds)
print(f" Model training complete. Precision on test data: {precision:.2f}")

# Save the trained model to a file
model_filename = "stock_xgb_model.joblib"
joblib.dump(model, model_filename)

print(f" Model saved as '{model_filename}'. You can now run ai_finance.py.")
