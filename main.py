# stock_analysis_free.py

import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from googlesearch import search
import requests
from bs4 import BeautifulSoup

# -------------------------------
# -------------------------------
def get_ticker(company_name):
    # You can add a dictionary of popular companies if search fails
    search_results = search(f"{company_name} stock ticker", num_results=1)
    if search_results:
        # Try to extract ticker from URL (works for Yahoo Finance links)
        for result in search_results:
            if "finance.yahoo.com/quote/" in result:
                ticker = result.split("/quote/")[1].split("?")[0]
                return ticker.upper()
    return None

# -------------------------------
# Step 2: Scrape recent news about the company
# -------------------------------
def get_recent_news(company_name, max_articles=3):
    query = f"{company_name} stock fell today"
    news_links = search(query, num_results=max_articles)
    news_texts = []

    for link in news_links:
        try:
            page = requests.get(link)
            soup = BeautifulSoup(page.content, 'html.parser')
            text = ' '.join([p.get_text() for p in soup.find_all('p')])
            news_texts.append(text)
        except:
            continue

    return news_texts

# -------------------------------
# -------------------------------
def get_fundamentals(ticker):
    stock = yf.Ticker(ticker)

    info = stock.info
    market_cap = info.get("marketCap", 0)
    book_value = info.get("bookValue", 0)
    price = info.get("regularMarketPrice", 0)
    previous_close = info.get("previousClose", 0)
    percent_change = round(((price - previous_close) / previous_close) * 100, 2) if previous_close else 0

    return {
        "Market Cap": market_cap,
        "Book Value": book_value,
        "Price": price,
        "Previous Close": previous_close,
        "Percentage Change": percent_change
    }

# -------------------------------
# -------------------------------
def get_stock_rating(ticker, csv_file="StockRatings.csv"):
    try:
        df = pd.read_csv(csv_file)
        row = df[df['Ticker'] == ticker]
        if not row.empty:
            return row.iloc[0]['Overall Rating']
        else:
            return "N/A"
    except:
        return "N/A"

# -------------------------------
# Display results
# -------------------------------
def analyze_company(company_name):
    print(f"\nAnalyzing {company_name}...\n")

    ticker = get_ticker(company_name)
    if not ticker:
        print("Ticker not found. Please check the company name.")
        return

    fundamentals = get_fundamentals(ticker)
    news = get_recent_news(company_name)
    rating = get_stock_rating(ticker)

    print(f"Company: {company_name} ({ticker})")
    print(f"Current Price: ${fundamentals['Price']}")
    print(f"Percentage Change Today: {fundamentals['Percentage Change']}%")
    print(f"Market Cap: ${fundamentals['Market Cap']}")
    print(f"Book Value: ${fundamentals['Book Value']}")
    print(f"Overall Stock Rating: {rating}/100\n")

    print("Recent news / reasons for drop:")
    if news:
        for i, article in enumerate(news, 1):
            print(f"{i}. {article[:300]}...")  # Print first 300 chars
    else:
        print("No news found.")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    company_name = input("Enter company name: ").strip()
    analyze_company(company_name)
