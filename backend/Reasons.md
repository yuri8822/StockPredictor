# Feature Selection for Next-Day Stock Price Prediction

## Structured Features Selection

The code implements a carefully selected set of structured features based on established financial analysis principles:

**Price-Based Features:**
- **Daily Returns (`Return`)**: The percentage change in adjusted closing price captures short-term momentum and is fundamental to understanding price dynamics. Returns are more stationary than raw prices, making them ideal for time series prediction.
- **Moving Averages (MA5, MA10)**: These trend-following indicators smooth out price noise and capture underlying directional movement. The 5-day MA captures short-term trends while the 10-day MA provides slightly longer-term context, creating a multi-timeframe perspective essential for next-day prediction.

**Volatility Measures:**
- **5-Day Rolling Volatility (`Volatility_5d`)**: Computed as the standard deviation of 5-day returns, this feature captures recent market uncertainty and risk. High volatility periods often precede significant price movements, making this crucial for next-day predictions.

**Volume and OHLC Data**: The code preserves Open, High, Low, Close, and Volume data, which contain intraday trading information and market participation levels that influence future price movements.

## Unstructured Features Selection

**News Sentiment Analysis:**
- **Aggregated Headlines**: Multiple daily headlines are concatenated to capture comprehensive market sentiment for each trading day.
- **Sentiment Polarity**: Using TextBlob's sentiment analysis, headlines are converted into numerical sentiment scores (-1 to +1). This captures market psychology and news-driven sentiment that often drives next-day price reactions.

## Sufficiency for Next-Day Prediction

This minimal feature set is sufficient for next-day price prediction for several reasons:

1. **Efficient Market Hypothesis Considerations**: In semi-strong efficient markets, only the most recent price information and new information (news) should predict future prices. Our features capture both recent price dynamics and fresh news sentiment.

2. **Short Prediction Horizon**: For next-day predictions, long-term fundamental indicators become less relevant compared to recent momentum, volatility, and immediate market sentiment captured by news.

3. **Balanced Information Types**: The combination covers three key dimensions:
   - **Technical**: Price momentum and volatility patterns
   - **Behavioral**: Market sentiment from news
   - **Structural**: Trading volume and intraday price ranges

4. **Computational Efficiency**: A minimal feature set reduces overfitting risk and computational complexity while maintaining predictive power for short-term forecasts.

5. **Real-time Availability**: All features can be computed in real-time using readily available market data and news feeds, making the model practically deployable.

## Limitations and Future Enhancements

While sufficient for basic next-day prediction, this feature set could be enhanced with:
- Market microstructure features (bid-ask spreads, order book data)
- Sector-specific news sentiment
- Correlation features with market indices
- Economic calendar events

## References

1. Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383-417.

2. Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not follow random walks: Evidence from a simple specification test. *The Review of Financial Studies*, 1(1), 41-66.

3. Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *The Journal of Finance*, 62(3), 1139-1168.

4. Bollen, J., Mao, H., & Zeng, X. (2011). Twitter mood predicts the stock market. *Journal of Computational Science*, 2(1), 1-8.