# main.py

import yfinance as yf
import requests
import gradio as gr
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import re
import joblib # ADDED: To load the model

# Machine Learning Imports
from xgboost import XGBClassifier
# We don't need train_test_split or precision_score here anymore
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import precision_score

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic.v1 import Field, BaseModel

# --- Gemini Safety Settings ---
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# -------------------------
# Load API Keys
# -------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# -------------------------
# Load Pre-trained ML Model
# (Run train_model.py first to create this file)
# -------------------------
MODEL_FILE = "stock_xgb_model.joblib"
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model file '{MODEL_FILE}' not found. "
        "Please run train_model.py first to create it."
    )
try:
    # Load the model ONCE when the app starts
    PRE_TRAINED_MODEL = joblib.load(MODEL_FILE)
    print(f"Pre-trained model '{MODEL_FILE}' loaded successfully.")
except Exception as e:
    raise Exception(f"Error loading model: {e}")

# -------------------------
# LangChain LLM Setup
# -------------------------
safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", # Using Flash for speed
    temperature=0.3,
    max_output_tokens=1024,
    google_api_key=GEMINI_API_KEY,
    safety_settings=safety_settings
)

# ============================================================
# XGBOOST PREDICTOR (The "Quant")
# ============================================================

class StockPredictor:
    def __init__(self, ticker):
        self.ticker = ticker
        # Use the globally loaded, pre-trained model
        self.model = PRE_TRAINED_MODEL
        self.data = None

    def fetch_data(self, period="2y"):
        # Fetch data for the specified period
        stock = yf.Ticker(self.ticker)
        df = stock.history(period=period)
        if df.empty:
            return False
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        self.data = df
        return True

    def add_technical_indicators(self, df):
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
        
        # 4. MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        
        return df

    # RENAMED: from train_and_predict -> get_prediction
    # REMOVED: All training logic
    def get_prediction(self):
        if self.data is None: return None
        
        df = self.add_technical_indicators(self.data)
        df = df.dropna() # Remove rows with NaN from rolling windows
        
        if df.empty: return None

        # Features for the model
        features = ["SMA_10", "SMA_50", "RSI", "Volatility", "MACD", "Open", "High", "Low", "Close", "Volume"]
        
        # We only need the most recent day to make a prediction
        test_row = df.iloc[[-1]]

        # --- NO TRAINING ---
        # We just predict using the pre-trained model
        preds = self.model.predict_proba(test_row[features])
        prob_buy = preds[0][1] # Probability of "Up"

        latest = test_row.iloc[0]
        
        signal = "HOLD/NEUTRAL"
        confidence = prob_buy

        if prob_buy > 0.55:
            signal = "BUY"
            confidence = prob_buy
        elif prob_buy < 0.45:
            signal = "SELL/AVOID"
            confidence = 1.0 - prob_buy
        else:
            confidence = 1.0 - abs(prob_buy - 0.5)

        return {
            "signal": signal,
            "confidence": round(confidence * 100, 1),
            "raw_buy_prob": prob_buy,
            "rsi": round(latest["RSI"], 2),
            "close": round(latest["Close"], 2),
            "sma_50": round(latest["SMA_50"], 2),
            "macd": round(latest["MACD"], 2)
        }

# ============================================================
# HYBRID STOCK ENGINE (Data Fetching Helpers)
# ============================================================

def detect_market(ticker):
    return "INDIA" if ticker.upper().endswith((".NS", ".BO")) else "US"

# ----------------------------------------------------------------
# REMOVED get_stock_history() - We will reuse predictor.data
# ----------------------------------------------------------------

# ============================================================
# LANGCHAIN HANDLERS (No changes needed)
# ============================================================

# --- Chain 1: Explain the Signal ---
hybrid_template = """
You are a senior financial analyst.
An XGBoost Machine Learning model has just analyzed the technical indicators for {ticker}.

Here is the data from the model:
- **Current Price:** {price}
- **ML Signal:** {ml_signal}
- **Model Confidence:** {confidence}%
- **RSI (14-day):** {rsi} (Over 70=Overbought, Under 30=Oversold)
- **MACD:** {macd}
- **50-Day Moving Avg:** {sma_50}

Your Task:
Write a 3-4 sentence explanation of this signal for a user.
1. Interpret the RSI and MACD in relation to the price.
2. Explain why the model might be suggesting {ml_signal} based on these numbers.
3. Add a disclaimer that ML models are probabilistic, not prophetic.

Format:
**Analysis:** [Your explanation here]
"""

hybrid_prompt = PromptTemplate(
    input_variables=["ticker", "price", "ml_signal", "confidence", "rsi", "macd", "sma_50"],
    template=hybrid_template
)
hybrid_chain = LLMChain(llm=llm, prompt=hybrid_prompt)

# --- Chain 2: Suggest Candidates (Return CSV) ---
candidate_template = """
The user is considering {ticker}, but our technical analysis model has flagged it as 'SELL/AVOID'.
I need to find a better alternative to run my ML model on.

List 3 strong alternative stock tickers in the same sector/industry.
Return ONLY a comma-separated list of tickers. Do not add text.

Example Output: MSFT, ORCL, IBM
"""
candidate_prompt = PromptTemplate(
    input_variables=["ticker"],
    template=candidate_template
)
candidate_chain = LLMChain(llm=llm, prompt=candidate_prompt)

# ============================================================
# MAIN AGENT CONTROLLER
# ============================================================

def financial_agent(ticker_input):
    ticker = ticker_input.strip().upper()
    
    # 1. Run the XGBoost Model (Main Stock)
    predictor = StockPredictor(ticker)
    # Fetch 2 years for the main ticker (for charting)
    data_fetched = predictor.fetch_data(period="2y")
    
    if not data_fetched:
        return f"Could not fetch historical data for {ticker}.", None

    # Use the FAST prediction function
    ml_result = predictor.get_prediction()
    
    if not ml_result:
        return f"Not enough data to run XGBoost analysis for {ticker}.", None

    # 2. Pass ML Results to LangChain LLM (Explanation)
    try:
        response = hybrid_chain.run(
            ticker=ticker,
            price=ml_result["close"],
            ml_signal=ml_result["signal"],
            confidence=ml_result["confidence"],
            rsi=ml_result["rsi"],
            macd=ml_result["macd"],
            sma_50=ml_result["sma_50"]
        )
        ai_text = response.strip()
    except Exception as e:
        ai_text = f"LangChain Error: {str(e)}"

    # 3. Intelligent Alternative Search (If Main is SELL/AVOID)
    alt_text = ""
    if "SELL" in ml_result["signal"] or "AVOID" in ml_result["signal"]:
        try:
            candidates_str = candidate_chain.run(ticker=ticker)
            candidates = [c.strip().upper() for c in candidates_str.split(",") if c.strip()]
            
            best_alt = None
            best_score = -1
            
            # This loop is now FAST because get_prediction() is fast
            for cand in candidates:
                if ".NS" in ticker and "." not in cand:
                    cand += ".NS"
                
                p = StockPredictor(cand)
                # Fetch only 100 days for alternatives (MUCH FASTER)
                if p.fetch_data(period="100d"):
                    # Use the FAST prediction function
                    res = p.get_prediction()
                    if res:
                        if res["raw_buy_prob"] > best_score:
                            best_score = res["raw_buy_prob"]
                            best_alt = (cand, res)
                        
                        if res["signal"] == "BUY" and res["confidence"] > 70:
                            break
            
            if best_alt:
                cand_ticker, res = best_alt
                icon = ""
                if res["signal"] == "BUY": icon = ""
                else: icon = ""
                
                msg = "Stronger technical setup found." if res["signal"] == "BUY" else "Best available alternative (sector is weak)."
                
                alt_text = f"""
\n---
**Verified Alternative: {cand_ticker}**
**Model Signal:** {res['signal']} (Confidence: {res['confidence']}%)
*Reasoning:* {msg} This stock was selected by checking {len(candidates)} AI-suggested candidates against the XGBoost model.
"""
            else:
                alt_text = "\n\n---\n*Could not verify a better alternative with sufficient data.*"

        except Exception as e:
            print(f"Alternative Search Error: {e}")
            alt_text = f"\n\n---\n*Error finding alternative: {str(e)}*"

    # 4. Format Output
    icon = ""
    if ml_result["signal"] == "BUY": icon = ""
    elif "SELL" in ml_result["signal"]: icon = ""

    out = f"""
### {icon} {ml_result['signal']} (Confidence: {ml_result['confidence']}%)
**Stock:** {ticker} | **Price:** {ml_result['close']}

**Technical Data (Used by XGBoost):**
* **RSI:** {ml_result['rsi']}
* **MACD:** {ml_result['macd']}
* **SMA (50-day):** {ml_result['sma_50']}

---
{ai_text}
{alt_text}
"""

    # 5. Chart Generation
    # OPTIMIZATION: Reuse the data we already fetched
    fig = None
    try:
        # Get the last 90 days from the data we already have
        history_df = predictor.data.iloc[-90:]
        
        if not history_df.empty:
            # Get dates and prices from the DataFrame
            dates = history_df['Date']
            prices = history_df['Close']
            
            sma_window = 50
            sma = prices.rolling(window=sma_window).mean()
            
            z = np.polyfit(range(len(prices)), prices, 1)
            p = np.poly1d(z)
            trend = p(range(len(prices)))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines", name="Price", line=dict(color='blue')))
            
            # The SMA trace will start later, which is correct
            fig.add_trace(go.Scatter(x=dates, y=sma, mode="lines", name="50-Day SMA", line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=dates, y=trend, mode="lines", name="Trend", line=dict(dash="dash", color='gray')))
            
            fig.update_layout(
                title=f"{ticker} - 90 Day Price Action",
                template="plotly_white",
                hovermode="x unified",
                yaxis_title="Price"
            )
    except Exception as e:
        print(f"Chart Error: {e}")

    return out, fig

# ============================================================
# GRADIO UI (No changes needed)
# ============================================================

with gr.Blocks(theme=gr.themes.Soft(), title="Hybrid AI Financial Agent") as iface:
    gr.Markdown("""
    # AI POWERED FINANCIAL AGENT USING LANGCHAIN
    """)

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(label="Stock Ticker", placeholder="AAPL, NVDA, RELIANCE.NS")
            btn = gr.Button("Analyze with AI + XGBoost", variant="primary")
            gr.Examples(["AAPL", "NVDA", "TSLA", "RELIANCE.NS", "TATASTEEL.NS"], inputs=inp)
            text = gr.Markdown(label="Hybrid Analysis")
        with gr.Column(scale=2):
            chart = gr.Plot(label="Price Chart")

    btn.click(financial_agent, inp, [text, chart])

if __name__ == "__main__":
    iface.launch()
