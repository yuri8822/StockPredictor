import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import argparse
import warnings
import feedparser
from textblob import TextBlob


# -------------------------------
# Get stock data
# -------------------------------
def get_stock_data(ticker, days):
    end = dt.datetime.today()
    start = end - dt.timedelta(days=days)

    df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False  # Ensure 'Adj Close' is present
        )
    
    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

        
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


# -------------------------------
# Get news via Yahoo Finance RSS
# -------------------------------
def get_news(ticker):
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(url)

        if not feed.entries:
            return pd.DataFrame(columns=["Date", "Headline", "Publisher"])

        news_data = []
        for entry in feed.entries:
            date = dt.datetime(*entry.published_parsed[:6]).date() if "published_parsed" in entry else None
            title = entry.title
            publisher = entry.get("source", {}).get("title", "Yahoo Finance")
            news_data.append({"Date": date, "Headline": title, "Publisher": publisher})

        return pd.DataFrame(news_data)

    except Exception as e:
        warnings.warn(f"Failed to get news for {ticker}: {e}")
        return pd.DataFrame(columns=["Date", "Headline", "Publisher"])


# -------------------------------
# Add technical indicators
# -------------------------------
def add_features(df):
    df["Return"] = df["Adj Close"].pct_change()
    df["Volatility_5d"] = df["Return"].rolling(window=5).std()
    df["MA5"] = df["Adj Close"].rolling(window=5).mean()
    df["MA10"] = df["Adj Close"].rolling(window=10).mean()
    return df


# -------------------------------
# Sentiment analysis
# -------------------------------
def analyze_sentiment(text):
    if pd.isna(text) or not text.strip():
        return 0.0
    return TextBlob(text).sentiment.polarity


# -------------------------------
# Merge stock + news
# -------------------------------
def merge_stock_news(df, news_df):
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    news_df["Date"] = pd.to_datetime(news_df["Date"]).dt.date

    merged = pd.merge(df, news_df, on="Date", how="left")

    # Group by Date so headlines are aligned correctly
    merged = (
        merged.groupby("Date")
        .agg({
            "Adj Close": "first",
            "Close": "first",
            "High": "first",
            "Low": "first",
            "Open": "first",
            "Volume": "first",
            "Return": "first",
            "Volatility_5d": "first",
            "MA5": "first",
            "MA10": "first",
            "Headline": lambda x: " | ".join(x.dropna().astype(str).unique()),
            "Publisher": lambda x: " | ".join(x.dropna().astype(str).unique())
        })
        .reset_index()
    )

    # Add sentiment
    merged["Sentiment"] = merged["Headline"].apply(analyze_sentiment)

    return merged


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", type=str, default="NASDAQ", help="Stock exchange (unused, just metadata)")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol, e.g., AAPL")
    parser.add_argument("--days", type=int, default=30, help="Days back for stock data")
    args = parser.parse_args()

    print(f"[INFO] Exchange: {args.exchange}")
    print(f"[INFO] Ticker: {args.ticker}")
    print(f"[INFO] Days back: {args.days}")

    df = get_stock_data(args.ticker, args.days)
    df = add_features(df)
    news_df = get_news(args.ticker)
    curated = merge_stock_news(df, news_df)

    filename = f"{args.ticker}_curated_dataset.csv"
    curated.to_csv(filename, index=False)

    print(f"[INFO] Saved curated dataset to: {filename}\n")
    print("Sample (first 5 rows):")
    print(curated.head())


if __name__ == "__main__":
    main()
