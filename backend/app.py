import streamlit as st
import requests
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

st.set_page_config(page_title="AI Financial Forecast", layout="centered", initial_sidebar_state="collapsed")

st.title("📈 InsightInvest Stock Forecast Dashboard")
st.markdown("Get instant, AI-driven investment insights and price forecasts, with real news sentiment.")

symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT):", "AAPL")

if st.button("Generate Forecast"):
    with st.spinner("Fetching AI insights..."):
        try:
            url = f"http://127.0.0.1:8000/forecast/{symbol.upper()}"
            response = requests.get(url, timeout=120)
            
            if response.status_code != 200:
                st.error(f"⚠️ Backend error: {response.status_code}\n{response.text}")
                st.stop()
            
            data = response.json()
            
            if 'detail' in data:
                st.error(f"⚠️ Error: {data['detail']}")
                st.stop()
            
            # --- Forecast Plot ---
            st.subheader(f"Forecast for {symbol.upper()}")
            plot_key = data.get("plot") or (data.get("visualization") or {}).get("chart")
            if plot_key:
                img_bytes = base64.b64decode(plot_key)
                st.image(io.BytesIO(img_bytes), caption="Forecast Plot")

            # --- Professional Report ---
            report = data.get("investment_report") or data.get("report") or "No report available."
            st.subheader("📘 Investment Report")
            st.markdown(report)

            # --- Market Sentiment / Summary ---
            st.subheader("Market Summary / Sentiment Analysis")
            sentiment = (data.get("market_sentiment") or {}).get("analysis") or data.get("sentiment") or {}
            if sentiment:
                sentiment_desc = sentiment.get("sentiment", "neutral").capitalize()
                sentiment_score = sentiment.get("sentiment_score", 0.5)
                sentiment_conf = sentiment.get("confidence", 0)
                sent_dist = sentiment.get("sentiment_distribution", {})
                pos = sent_dist.get("positive", 0)
                neu = sent_dist.get("neutral", 0)
                neg = sent_dist.get("negative", 0)
                total = pos + neu + neg

                st.markdown(
                    f"**Market Sentiment:** {sentiment_desc} &ensp; | &ensp; "
                    f"Positive: {pos} &ensp; Neutral: {neu} &ensp; Negative: {neg} &ensp; (Total articles: {total})"
                )
                st.markdown(
                    f"**Sentiment Score:** {sentiment_score:.2f} &ensp; | &ensp; "
                    f"Confidence: {sentiment_conf:.0%}"
                )

                # Show sentiment as a bar chart
                st.subheader("Sentiment Breakdown")
                st.bar_chart({
                    "Positive": [pos],
                    "Neutral": [neu],
                    "Negative": [neg]
                })

                # Show a sentiment gauge (progress bar)
                st.subheader("Overall Sentiment Gauge")
                st.progress(sentiment_score)
                
                # Emoji/Color summary
                if sentiment_desc == "Positive":
                    st.success("🟢 Market is Optimistic")
                elif sentiment_desc == "Negative":
                    st.error("🔴 Market is Pessimistic")
                else:
                    st.info("🟡 Market is Neutral or Uncertain")

            else:
                st.write("No sentiment analysis available.")

            # --- Predicted Prices (line chart) ---
            forecast = data.get("price_forecast") or data.get("forecast")
            if forecast and "mean" in forecast:
                st.subheader("Predicted Prices")
                st.line_chart(np.array(forecast["mean"]))
            else:
                st.write("No forecast data available.")

        except requests.ConnectionError:
            st.error("⚠️ Backend server is not running. Please start your FastAPI backend on http://127.0.0.1:8000")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Instructions for deployment/testing
st.markdown("""
---
#### Instructions
1. Make sure your FastAPI backend is running at `http://127.0.0.1:8000` before clicking "Generate Forecast".
2. The analysis combines live stock data, AI-powered news sentiment, and forecast modeling.
""")
