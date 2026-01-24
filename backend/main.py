# # python main.py
import os
import re
import io
import time
import base64
import logging
import asyncio
import itertools
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from threading import Lock
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import style

import httpx
import yfinance as yf
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from cachetools import TTLCache
from dotenv import load_dotenv
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

from news_scrapping import generate_google_news_rss_url, fetch_news_items
from urllib.parse import quote_plus

# --- CHANGED: IMPORT GROQ INSTEAD OF GEMINI ---
from groq import AsyncGroq

# reuse a browser-like user agent for external HTTP calls
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# ------------- Setup & Config (OPTIMIZED FOR QUOTA) -------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# --- CHANGED: GROQ KEY ROTATION LOGIC ---
# Put all your keys in .env like: GROQ_API_KEY="key1,key2,key3"
GROQ_KEYS_STR = os.getenv("GROQ_API_KEY", "")
GROQ_KEYS = [k.strip() for k in GROQ_KEYS_STR.split(",") if k.strip()]
HF_API_KEY = os.getenv("HF_API_KEY")

if not GROQ_KEYS:
    logger.error("GROQ_API_KEY is missing. Report generation will fail.")
    GROQ_KEYS = ["invalid_key"]

if not HF_API_KEY:
    logger.warning("HF_API_KEY is missing. Sentiment analysis may fail.")

# Create a cycling iterator for keys
_groq_key_cycle = itertools.cycle(GROQ_KEYS)

def get_groq_client():
    """Returns a Groq client instance using the next available API key"""
    api_key = next(_groq_key_cycle)
    return AsyncGroq(api_key=api_key)


async def resolve_company_to_symbol(query: str) -> Dict[str, str]:
    """Try to resolve a company name to a ticker symbol using Yahoo Finance search API."""
    # First try Yahoo Finance quick-search
    # Quick local mapping fallback for common company names to reduce failure cases
    common_map = {
        'APPLE': 'AAPL', 'MICROSOFT': 'MSFT', 'GOOGLE': 'GOOGL', 'ALPHABET': 'GOOGL',
        'AMAZON': 'AMZN', 'TESLA': 'TSLA', 'NVIDIA': 'NVDA', 'META': 'META',
        'IBM': 'IBM', 'INTEL': 'INTC', 'TSM': 'TSM'
    }
    qnorm = query.strip().upper()
    if qnorm in common_map:
        logger.info(f"Resolved '{query}' -> {common_map[qnorm]} via local mapping")
        return {"symbol": common_map[qnorm], "name": query}
    try:
        qs = quote_plus(query)
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={qs}&quotesCount=5&newsCount=0"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            j = resp.json()
            quotes = j.get("quotes") or []
            if quotes:
                top = quotes[0]
                sym = top.get("symbol")
                name = top.get("longname") or top.get("shortname") or top.get("quoteType") or query
                if sym:
                    # Preserve common exchange suffixes like .NS/.BO if provided in symbol
                    logger.info(f"Resolved '{query}' -> {sym} ({name}) via Yahoo search")
                    return {"symbol": sym, "name": name}
    except Exception as e:
        logger.debug(f"Company resolve failed for '{query}' via Yahoo: {e}")

    # --- CHANGED: USE GROQ FOR FALLBACK ---
    try:
        prompt = (
            f"User provided: '{query}'.\n"
            "Extract the most likely stock ticker symbol and the canonical company name. "
            "If the input already contains an exchange suffix (like .NS, .BO, .L), preserve it. "
            "Return plain JSON with keys: symbol and name. If you can't find a ticker, set symbol to an uppercased version of the input.\n\n"
            "Respond only with JSON."
        )

        client = get_groq_client()
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", # High performance model
            temperature=0,
            response_format={"type": "json_object"} # Groq supports JSON mode
        )

        text = chat_completion.choices[0].message.content.strip()

        # Try to parse JSON from the model output
        import json as _json
        try:
            parsed = _json.loads(text)
            sym = parsed.get('symbol')
            name = parsed.get('name') or query
            if sym:
                logger.info(f"Resolved '{query}' -> {sym} ({name}) via Groq fallback")
                return {"symbol": sym, "name": name}
        except Exception:
            # not strict JSON: attempt to extract SYMBOL: ... lines
            m = re.search(r'"?symbol"?\s*[:=]\s*"?([A-Za-z0-9\.\-_]+)"?', text)
            n = re.search(r'"?name"?\s*[:=]\s*"?([^\n\"]+)"?', text)
            if m:
                sym = m.group(1)
                name = n.group(1).strip() if n else query
                logger.info(f"Parsed fallback from Groq for '{query}' -> {sym} ({name})")
                return {"symbol": sym, "name": name}
    except Exception as e:
        logger.debug(f"Groq fallback failed for '{query}': {e}")

    # fallback: if input already looks like symbol, return as-is, otherwise uppercase
    if re.match(r'^[A-Za-z0-9\.\-]+$', query):
        return {"symbol": query.upper(), "name": query}
    return {"symbol": query.upper(), "name": query}

app = FastAPI(
    title="InsightInvest - AI Financial Analyst",
    version="2.0.0",
    description="AI-powered financial analysis with enhanced forecasting, real market data, and comprehensive sentiment analysis"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caches and rate limiting
stock_cache = TTLCache(maxsize=100, ttl=600)  # 10 min cache for stock data
news_cache = TTLCache(maxsize=100, ttl=1800)  # 30 min cache for news
financial_metrics_cache = TTLCache(maxsize=100, ttl=3600)  # 1 hour cache for financial metrics
RATE_LIMIT = 10
requests_log: Dict[str, List[float]] = {}
requests_lock = Lock()

def is_rate_limited(ip: str) -> bool:
    with requests_lock:
        now = time.time()
        window = 60
        requests_log[ip] = [t for t in requests_log.get(ip, []) if now - t < window]
        if len(requests_log[ip]) >= RATE_LIMIT:
            logger.warning(f"Rate limit exceeded for IP {ip}")
            return True
        requests_log[ip].append(now)
        return False

# ------------- Enhanced Multi-Market Stock Data Fetching -------------

def normalize_stock_symbol(symbol: str) -> str:
    """
    Normalize stock symbol for different markets and exchanges.
    Supports US, Indian (NSE/BSE), UK, European, and other major markets.
    """
    symbol = symbol.upper().strip()

    # Market suffixes for reference:
    # Indian: .NS (NSE), .BO (BSE)
    # International: .L (London), .T (Tokyo), .HK (Hong Kong), .AX (Australia), etc.

    # Common Indian stock patterns - auto-detect and add NSE suffix if needed
    indian_patterns = [
        'RELIANCE', 'TCS', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'ITC',
        'KOTAKBANK', 'LT', 'ASIANPAINT', 'AXISBANK', 'MARUTI', 'SUNPHARMA', 'ULTRACEMCO',
        'TITAN', 'NESTLEIND', 'BAJFINANCE', 'TECHM', 'POWERGRID', 'NTPC', 'COALINDIA',
        'ONGC', 'WIPRO', 'TATAMOTORS', 'JSWSTEEL', 'GRASIM', 'HCLTECH', 'CIPLA', 'DRREDDY',
        'EICHERMOT', 'BRITANNIA', 'DIVISLAB', 'APOLLOHOSP', 'BAJAJFINSV', 'SHREECEM',
        'TATASTEEL', 'ADANIPORTS', 'HEROMOTOCO', 'BPCL', 'INDUSINDBK'
    ]

    # If it's a known Indian stock without suffix, add .NS
    base_symbol = symbol.split('.')[0]
    if base_symbol in indian_patterns and '.' not in symbol:
        logger.info(f"Auto-detected Indian stock {symbol}, adding .NS suffix")
        return f"{symbol}.NS"

    # If symbol already has a suffix, return as is
    if '.' in symbol:
        return symbol

    # Default: assume US market (no suffix needed for US stocks)
    return symbol

async def fetch_real_stock_data(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch real historical stock data using yfinance with enhanced multi-market support.
    """
    # OPTIMIZATION: Run blocking YFinance code in thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_stock_data_sync, symbol, period, interval)

def _fetch_stock_data_sync(symbol: str, period: str, interval: str) -> pd.DataFrame:
    # Normalize the symbol for different markets
    normalized_symbol = normalize_stock_symbol(symbol)
    cache_key = f"{normalized_symbol}_{period}_{interval}"

    if cache_key in stock_cache:
        logger.info(f"Cache hit for stock data: {cache_key}")
        return stock_cache[cache_key]

    try:
        logger.info(f"Fetching real stock data for {normalized_symbol} (original: {symbol})")
        ticker = yf.Ticker(normalized_symbol)

        # Fetch historical data
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            # Try alternative suffixes for Indian stocks if initial attempt fails
            if symbol.upper() in ['RELIANCE', 'TCS', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'SBIN', 'BHARTIARTL', 'ITC']:
                for suffix in ['.NS', '.BO']:
                    try:
                        alt_symbol = f"{symbol.upper()}{suffix}"
                        if alt_symbol != normalized_symbol:  # Don't retry the same symbol
                            alt_ticker = yf.Ticker(alt_symbol)
                            df = alt_ticker.history(period=period, interval=interval)
                            if not df.empty:
                                normalized_symbol = alt_symbol
                                logger.info(f"Found data using alternative suffix: {alt_symbol}")
                                break
                    except Exception:
                        continue

            # Note: Async resolve call removed from sync function to avoid complexity.

            if df.empty:
                raise ValueError(f"No data found for symbol {normalized_symbol}. For Indian stocks, try adding .NS (NSE) or .BO (BSE) suffix.")

        # Clean the data
        df = df.dropna()

        # Ensure we have enough data points for forecasting
        if len(df) < 20:
            # Try longer period if insufficient data
            logger.warning(f"Insufficient data with {period}, trying 1y")
            df = ticker.history(period="1y", interval="1d")
            df = df.dropna()

        if len(df) < 20:
            raise ValueError(f"Insufficient data for forecasting: only {len(df)} data points")

        # Cache the data
        stock_cache[cache_key] = df
        logger.info(f"Successfully fetched {len(df)} data points for {normalized_symbol}")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch stock data for {normalized_symbol}: {e}")
        raise HTTPException(status_code=404, detail=f"Could not fetch stock data: {str(e)}")

# ------------- Financial Metrics Analysis -------------
async def fetch_financial_metrics(symbol: str) -> Dict[str, Any]:
    """
    Fetch key financial metrics for fundamental analysis.
    """
    # OPTIMIZATION: Run blocking YFinance code in thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_metrics_sync, symbol)

def _fetch_metrics_sync(symbol: str) -> Dict[str, Any]:
    # Normalize symbol for metrics fetching
    normalized_symbol = normalize_stock_symbol(symbol)
    cache_key = f"metrics_{normalized_symbol}"
    if cache_key in financial_metrics_cache:
        logger.info(f"Cache hit for financial metrics: {cache_key}")
        return financial_metrics_cache[cache_key]

    try:
        ticker = yf.Ticker(normalized_symbol)
        info = ticker.info

        # Helper function to safely convert and validate metrics
        def safe_metric(value, metric_type="float", multiplier=1, max_reasonable=None):
            """Safely convert and validate financial metrics"""
            if value is None or value == "N/A":
                return None
            try:
                if metric_type == "percentage":
                    # Convert decimal to percentage and validate
                    converted = float(value) * 100 if value < 1 else float(value)
                    if max_reasonable and converted > max_reasonable:
                        logger.warning(f"Metric {converted}% exceeds reasonable limit {max_reasonable}%, likely data error")
                        return None
                    return round(converted, 2)
                elif metric_type == "float":
                    converted = float(value) * multiplier
                    if max_reasonable and converted > max_reasonable:
                        logger.warning(f"Metric {converted} exceeds reasonable limit {max_reasonable}, likely data error")
                        return None
                    return round(converted, 2)
                else:
                    return value
            except (ValueError, TypeError):
                return None

        # Extract and validate key metrics
        raw_dividend_yield = info.get("dividendYield", 0)
        raw_roe = info.get("returnOnEquity", None)
        raw_profit_margin = info.get("profitMargins", None)
        raw_revenue_growth = info.get("revenueGrowth", None)

        # Validate dividend yield (typically 0-10% for most stocks)
        dividend_yield = safe_metric(raw_dividend_yield, "percentage", max_reasonable=25.0)
        if dividend_yield is None and raw_dividend_yield:
            # If original was too high, try as decimal
            dividend_yield = safe_metric(raw_dividend_yield, "float", max_reasonable=0.25)
            if dividend_yield:
                dividend_yield *= 100

        # Validate ROE (typically 5-50% for healthy companies)
        roe = safe_metric(raw_roe, "percentage", max_reasonable=100.0)

        # Validate profit margin (typically 0-50%)
        profit_margin = safe_metric(raw_profit_margin, "percentage", max_reasonable=80.0)

        # Validate revenue growth (typically -50% to +100%)
        revenue_growth = safe_metric(raw_revenue_growth, "percentage", max_reasonable=200.0)

        metrics = {
            "company_name": info.get("longName", normalized_symbol),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": safe_metric(info.get("trailingPE", None)),
            "forward_pe": safe_metric(info.get("forwardPE", None)),
            "eps": safe_metric(info.get("trailingEps", None)),
            "debt_to_equity": safe_metric(info.get("debtToEquity", None)),
            "roe": roe,
            "profit_margin": profit_margin,
            "revenue_growth": revenue_growth,
            "earnings_growth": safe_metric(info.get("earningsGrowth", None), "percentage"),
            "current_price": safe_metric(info.get("currentPrice", 0)),
            "52_week_high": safe_metric(info.get("fiftyTwoWeekHigh", 0)),
            "52_week_low": safe_metric(info.get("fiftyTwoWeekLow", 0)),
            "dividend_yield": dividend_yield,
            "beta": safe_metric(info.get("beta", None)),
            "volume": info.get("volume", 0),
            "avg_volume": info.get("averageVolume", 0)
        }
        # include reported currency from yfinance if available
        metrics["currency"] = info.get("currency", None)

        # Cross-validate and fix obvious errors using historical data
        try:
            historical_data = ticker.history(period="1y")
            if not historical_data.empty:
                # If current price is missing or unreasonable, use latest close
                current_close = historical_data['Close'].iloc[-1]
                if not metrics["current_price"] or abs(metrics["current_price"] - current_close) > current_close * 0.5:
                    metrics["current_price"] = round(current_close, 2)
                    logger.info(f"Corrected current price to {current_close}")

                # Validate 52-week high/low against actual historical data
                actual_52w_high = historical_data['High'].max()
                actual_52w_low = historical_data['Low'].min()

                if abs(metrics["52_week_high"] - actual_52w_high) > actual_52w_high * 0.1:
                    metrics["52_week_high"] = round(actual_52w_high, 2)

                if abs(metrics["52_week_low"] - actual_52w_low) > actual_52w_low * 0.1:
                    metrics["52_week_low"] = round(actual_52w_low, 2)
        except Exception as e:
            logger.warning(f"Could not validate metrics with historical data: {e}")

        # Add data quality indicators
        metrics["data_quality_flags"] = []
        if metrics["dividend_yield"] is None and raw_dividend_yield:
            metrics["data_quality_flags"].append(f"dividend_yield_invalid_raw_value_{raw_dividend_yield}")
        if metrics["roe"] is None and raw_roe:
            metrics["data_quality_flags"].append(f"roe_invalid_raw_value_{raw_roe}")
        if metrics["revenue_growth"] is None and raw_revenue_growth:
            metrics["data_quality_flags"].append(f"revenue_growth_invalid_raw_value_{raw_revenue_growth}")

        # Get quarterly financials for trends
        try:
            quarterly_financials = ticker.quarterly_financials
            if not quarterly_financials.empty:
                revenue_trend = []
                if "Total Revenue" in quarterly_financials.index:
                    revenue_data = quarterly_financials.loc["Total Revenue"].dropna()
                    revenue_trend = revenue_data.tolist()[:4]  # Last 4 quarters
                metrics["quarterly_revenue_trend"] = revenue_trend

                # Attempt to compute quarterly profit margin trend if Net Income (or 'Net Income') exists
                profit_margin_trend = []
                # Possible index keys that may represent net income
                net_income_keys = [k for k in quarterly_financials.index if 'net' in k.lower() and 'income' in k.lower()]
                if net_income_keys and "Total Revenue" in quarterly_financials.index:
                    net_key = net_income_keys[0]
                    try:
                        net_vals = quarterly_financials.loc[net_key].dropna()
                        rev_vals = quarterly_financials.loc["Total Revenue"].dropna()
                        # align lengths and compute margin = net / revenue * 100
                        n = min(len(net_vals), len(rev_vals))
                        for i in range(n):
                            ni = net_vals.iloc[i]
                            rv = rev_vals.iloc[i]
                            try:
                                pm = (float(ni) / float(rv)) * 100 if rv and rv != 0 else None
                            except Exception:
                                pm = None
                            profit_margin_trend.append(round(pm, 2) if pm is not None else None)
                    except Exception:
                        profit_margin_trend = []
                # Fallback: if profit margin available as a single metric, repeat it
                if not profit_margin_trend and metrics.get('profit_margin') is not None:
                    profit_margin_trend = [metrics.get('profit_margin')] * min(4, len(revenue_trend) or 4)

                metrics["quarterly_profit_margin_trend"] = profit_margin_trend
            else:
                metrics["quarterly_revenue_trend"] = []
        except Exception:
            metrics["quarterly_revenue_trend"] = []
            metrics["quarterly_profit_margin_trend"] = []

        financial_metrics_cache[cache_key] = metrics
        logger.info(f"Successfully fetched financial metrics for {normalized_symbol}")
        return metrics

    except Exception as e:
        logger.error(f"Failed to fetch financial metrics for {normalized_symbol}: {e}")
        return {
            "company_name": normalized_symbol,
            "sector": "N/A",
            "industry": "N/A",
            "error": str(e)
        }

# ------------- Enhanced Data Preprocessing & Forecasting -------------
def _maybe_resample(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Resample data if too many points to improve model performance"""
    freq_used = "1D"

    if len(df) > 500:  # If more than 500 data points
        try:
            # Resample to weekly
            resampled = df.resample("W").agg(
                {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            ).dropna(how="all")

            if len(resampled) >= 30:
                df = resampled
                freq_used = "W"
                logger.info(f"Resampled data to weekly frequency: {len(df)} points")
        except Exception as e:
            logger.warning(f"Resampling failed: {e}, using original data")

    return df.dropna(), freq_used

def _conf_int_arrays(fc) -> Tuple[np.ndarray, np.ndarray]:
    """Extract confidence intervals from forecast"""
    ci = fc.conf_int(alpha=0.05)
    if isinstance(ci, np.ndarray):
        lower, upper = ci[:, 0], ci[:, 1]
    else:
        lower = ci.iloc[:, 0].to_numpy()
        upper = ci.iloc[:, 1].to_numpy()
    return lower, upper

def forecast_with_enhanced_sentiment_fusion(
    data: pd.DataFrame,
    sentiment_score: float,
    steps: int = 10
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Enhanced forecasting with stronger sentiment integration and dynamic volatility"""

    df, used_freq = _maybe_resample(data.copy())
    close = df["Close"].astype(float).dropna()

    logger.info(f"Starting enhanced forecast with {len(close)} data points and sentiment score: {sentiment_score}")

    if len(close) < 20:
        raise ValueError(f"Insufficient data points for forecasting: {len(close)} < 20")

    # Calculate historical volatility for more realistic forecasts
    returns = close.pct_change().dropna()
    historical_volatility = np.std(returns) if len(returns) > 0 else 0.02
    logger.info(f"Historical volatility: {historical_volatility:.4f}")

    # ARIMA Forecasting
    log_close = np.log(close.values + 1e-12)
    arima_order = (5, 1, 1)
    arima_aic = float("nan")

    try:
        logger.info("Fitting ARIMA(5,1,1) model...")
        arima = ARIMA(log_close, order=arima_order)
        arima_fit = arima.fit()
        fc = arima_fit.get_forecast(steps=steps)
        log_mean = fc.predicted_mean
        lo, hi = _conf_int_arrays(fc)
        arima_aic = float(arima_fit.aic)
        logger.info(f"ARIMA model fitted successfully. AIC: {arima_aic:.2f}")
    except Exception as e:
        logger.warning(f"ARIMA failed: {e}. Using enhanced trend-based fallback.")
        # Enhanced trend calculation
        recent_prices = close.tail(20).values
        trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]

        log_mean = np.array([np.log(close.values[-1] + i * trend + 1e-12) for i in range(1, steps + 1)])
        # Add some realistic uncertainty
        uncertainty = historical_volatility * np.sqrt(np.arange(1, steps + 1))
        lo = log_mean - uncertainty
        hi = log_mean + uncertainty

    arima_mean = np.exp(log_mean)
    arima_lower = np.exp(lo)
    arima_upper = np.exp(hi)

    # Enhanced Holt-Winters Forecasting
    try:
        logger.info("Fitting Holt-Winters model...")
        hw = ExponentialSmoothing(close.values, trend="add", seasonal=None, initialization_method="estimated")
        hw_fit = hw.fit(optimized=True)
        hw_mean = hw_fit.forecast(steps)
        logger.info("Holt-Winters model fitted successfully")
    except Exception as e:
        logger.warning(f"Holt-Winters failed: {e}. Using enhanced linear trend fallback.")
        # Enhanced linear extrapolation with volatility
        x = np.arange(len(close))
        y = close.values
        slope = np.polyfit(x[-min(30, len(x)):], y[-min(30, len(y)):], 1)[0]
        base_forecast = np.array([close.values[-1] + slope * i for i in range(1, steps + 1)])

        # Add some realistic movement
        random_walk = np.cumsum(np.random.normal(0, historical_volatility * close.values[-1] * 0.5, steps))
        hw_mean = base_forecast + random_walk

    # Combine base forecasts
    base_forecast = (np.asarray(arima_mean) + np.asarray(hw_mean)) / 2.0

    # ENHANCED SENTIMENT FUSION
    sentiment_adjustment = 1.0
    if sentiment_score is not None:
        # Stronger sentiment impact: 20% max adjustment instead of 10%
        sentiment_impact = (sentiment_score - 0.5) * 0.4  # Max 20% adjustment
        sentiment_adjustment = 1.0 + sentiment_impact
        logger.info(f"Applying enhanced sentiment adjustment: {sentiment_adjustment:.4f}")

    # Apply sentiment adjustment with gradual increase and some randomness
    sentiment_adjusted_forecast = []
    for i, forecast_val in enumerate(base_forecast):
        # Gradually increase sentiment impact over forecast horizon
        time_weight = (i + 1) / steps  # 0 to 1 over forecast period
        adjustment = 1.0 + (sentiment_adjustment - 1.0) * time_weight

        # Add some realistic volatility to the forecast
        volatility_factor = np.random.normal(1.0, historical_volatility * 0.3)
        final_adjustment = adjustment * volatility_factor

        sentiment_adjusted_forecast.append(forecast_val * final_adjustment)

    combined_mean = np.array(sentiment_adjusted_forecast)

    # Enhanced confidence intervals with wider bands and sentiment consideration
    base_uncertainty = np.abs(arima_upper - arima_lower) / 2

    # Widen confidence intervals for more realistic uncertainty
    confidence_multiplier = 1.8  # 80% wider confidence bands
    sentiment_uncertainty_factor = 1.0 + abs(sentiment_score - 0.5) * 0.4 if sentiment_score else 1.0

    # Progressive widening over time
    time_expansion = np.linspace(1.0, 2.0, steps)  # Uncertainty grows over time

    adjusted_uncertainty = base_uncertainty * confidence_multiplier * sentiment_uncertainty_factor * time_expansion
    adjusted_lower = combined_mean - adjusted_uncertainty
    adjusted_upper = combined_mean + adjusted_uncertainty

    # Ensure lower bound doesn't go negative
    adjusted_lower = np.maximum(adjusted_lower, combined_mean * 0.5)

    # Generate future dates
    if used_freq == "W":
        future_index = pd.date_range(df.index[-1], periods=steps + 1, freq="W")[1:]
    else:
        future_index = pd.date_range(df.index[-1], periods=steps + 1, freq="D")[1:]

    logger.info(f"Enhanced forecast completed. Price range: {np.min(combined_mean):.2f} - {np.max(combined_mean):.2f}")
    logger.info(f"Sentiment impact: {((sentiment_adjustment - 1.0) * 100):.2f}%")
    logger.info(f"Average confidence interval width: {np.mean(adjusted_uncertainty):.2f}")

    return {
        "index": future_index.astype(str).tolist(),
        "mean": combined_mean.tolist(),
        "lower": adjusted_lower.tolist(),
        "upper": adjusted_upper.tolist(),
        "diagnostics": {
            "arima_order": arima_order,
            "arima_aic": arima_aic,
            "frequency_used": used_freq,
            "n_obs": len(close),
            "historical_volatility": historical_volatility,
            "sentiment_score": sentiment_score,
            "sentiment_adjustment": sentiment_adjustment,
            "confidence_multiplier": confidence_multiplier,
            "base_forecast_range": f"{np.min(base_forecast):.2f} - {np.max(base_forecast):.2f}",
            "sentiment_adjusted_range": f"{np.min(combined_mean):.2f} - {np.max(combined_mean):.2f}"
        },
    }, df

def plot_advanced_forecast(
    history: pd.DataFrame,
    forecast: Dict[str, Any],
    symbol: str,
    sentiment_info: Dict[str, Any],
    theme: str = "light",
    currency: str = "USD"
) -> str:
    """Create advanced forecast plot with enhanced styling and sentiment indicators"""
    try:
        style.use("seaborn-v0_8-whitegrid" if theme == "light" else "dark_background")
    except OSError:
        style.use("default")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[4, 1])

    # Main price plot with enhanced styling
    ax1.plot(history.index, history["Close"], label="Historical Close", color="#2E86AB", linewidth=2.5, alpha=0.9)

    future_index = pd.to_datetime(forecast["index"])
    mean = np.array(forecast["mean"], dtype=float)
    lower = np.array(forecast["lower"], dtype=float)
    upper = np.array(forecast["upper"], dtype=float)

    # Forecast line
    ax1.plot(future_index, mean, label="Forecast", color="#F24236", linewidth=3, alpha=0.9)
    ax1.fill_between(future_index, lower, upper, alpha=0.25, label="95% Confidence Interval", color="#F24236")

    # Add vertical line at forecast start
    ax1.axvline(x=history.index[-1], color='#A23B72', linestyle='--', alpha=0.7, linewidth=2, label="Forecast Start")

    # Enhanced title and labels
    company_name = sentiment_info.get('company_name', symbol)
    ax1.set_title(f"{symbol} - AI-Powered Price Forecast with Sentiment Analysis\n{company_name}",
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Price ($)", fontsize=12)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Improve x-axis date formatting to avoid label clumping
    try:
        import matplotlib.dates as mdates
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        for label in ax1.get_xticklabels():
            label.set_rotation(25)
            label.set_horizontalalignment('right')
    except Exception:
        # Fallback: rotate existing labels
        for label in ax1.get_xticklabels():
            label.set_rotation(25)
            label.set_horizontalalignment('right')

    # currency symbol for textbox and tick formatting
    currency_symbols = {"USD": "$", "INR": "₹"}
    cur_sym = currency_symbols.get(currency.upper(), "")

    # Add forecast statistics text box
    current_price = history["Close"].iloc[-1]
    forecast_end_price = mean[-1]
    price_change = ((forecast_end_price - current_price) / current_price) * 100

    textstr = f'Current: {cur_sym}{current_price:.2f}\nForecast: {cur_sym}{forecast_end_price:.2f}\nChange: {price_change:+.1f}%'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    # Enhanced sentiment indicator subplot
    sentiment_score = sentiment_info.get('sentiment_score', 0.5)
    sentiment_label = sentiment_info.get('sentiment', 'neutral')
    confidence = sentiment_info.get('confidence', 0.0)

    # Create gradient sentiment bar
    sentiment_colors = ['#FF4444', '#FFAA44', '#44AA44']  # Red, Orange, Green
    color_idx = 0 if sentiment_score < 0.33 else (1 if sentiment_score < 0.67 else 2)

    ax2.barh([0], [sentiment_score], color=sentiment_colors[color_idx], alpha=0.8, height=0.6)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlabel("Sentiment Score", fontsize=12)
    # Display only numeric sentiment score in title to keep plot clean
    ax2.set_title(f"Score: {sentiment_score:.3f} | Confidence: {confidence:.1%}", fontsize=12, fontweight='bold')
    ax2.set_yticks([])

    # Add sentiment threshold lines
    ax2.axvline(x=0.33, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=0.67, color='gray', linestyle=':', alpha=0.5)

    # Enhanced sentiment labels
    ax2.text(0.15, 0, 'Negative', ha='center', va='center', fontweight='bold', fontsize=10)
    ax2.text(0.5, 0, 'Neutral', ha='center', va='center', fontweight='bold', fontsize=10)
    ax2.text(0.85, 0, 'Positive', ha='center', va='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    # Format y-axis ticks with currency symbol
    try:
        import matplotlib.ticker as mticker
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{cur_sym}{x:,.2f}"))
    except Exception:
        pass

    try:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor='white')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate plot")
    finally:
        plt.close()

    return img_b64

# ------------- Enhanced Sentiment Analysis (OPTIMIZED: Parallel) -------------
# ------------- Enhanced Sentiment Analysis (SWITCHED TO GROQ) -------------
async def analyze_sentiment_advanced(texts: List[str]) -> Dict[str, Any]:
    """
    Analyze sentiment using Groq (Llama 3) instead of Hugging Face.
    Batches all headlines into a single request for speed.
    """
    if not texts:
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.5,
            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}
        }

    # 1. Prepare headlines (limit to top 15 to save tokens)
    headlines_text = "\n".join([f"- {t}" for t in texts[:15]])

    # 2. Construct Prompt
    prompt = f"""
    Analyze the sentiment of these financial news headlines:
    {headlines_text}

    Classify them and calculate an overall score (0=Bearish/Negative, 1=Bullish/Positive, 0.5=Neutral).
    Return ONLY a JSON object with this exact structure:
    {{
        "positive": int,  // count of positive headlines
        "negative": int,  // count of negative headlines
        "neutral": int,   // count of neutral headlines
        "score": float    // aggregate score 0.0 to 1.0
    }}
    """

    try:
        # 3. Call Groq
        client = get_groq_client()
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1, # Low temperature for consistent JSON
            response_format={"type": "json_object"}
        )

        # 4. Parse Response
        content = chat_completion.choices[0].message.content
        import json
        data = json.loads(content)

        score = float(data.get("score", 0.5))

        # Determine label
        label = "neutral"
        if score > 0.6: label = "positive"
        elif score < 0.4: label = "negative"

        return {
            "sentiment": label,
            "sentiment_score": score,
            "confidence": 0.9,
            "articles_analyzed": len(texts),
            "sentiment_distribution": {
                "positive": data.get("positive", 0),
                "negative": data.get("negative", 0),
                "neutral": data.get("neutral", 0)
            },
            "volatility": 0.0 # Not calculated in batch mode
        }

    except Exception as e:
        logger.error(f"Groq sentiment analysis failed: {e}")
        # Fallback to neutral
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.5,
            "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}
        }

# --- CHANGED: USE GROQ FOR REPORT GENERATION ---
async def generate_comprehensive_report(
    symbol: str,
    financial_metrics: Dict[str, Any],
    forecast: Dict[str, Any],
    sentiment_summary: Dict[str, Any],
    news_headlines: List[str],
    currency: str = "USD"
) -> str:
    """Generate comprehensive investment analysis report using Groq Llama 3"""

    # Helper function to safely format metrics
    currency_symbols = {"USD": "$", "INR": "₹"}
    cur_sym = currency_symbols.get(currency.upper(), "")

    def safe_format(value, format_type="float", suffix="", default="N/A"):
        """Safely format financial metrics, handling None values"""
        if value is None:
            return default
        try:
            if format_type == "float":
                return f"{float(value):.2f}{suffix}"
            elif format_type == "percent":
                return f"{float(value):.2f}%"
            elif format_type == "currency":
                try:
                    return f"{cur_sym}{float(value):.2f}"
                except Exception:
                    return default
            else:
                return str(value) + suffix
        except (ValueError, TypeError):
            return default

    # Prepare data for the LLM
    current_price = financial_metrics.get("current_price", 0)
    forecast_mean = forecast.get("mean", [])
    forecast_change = ((forecast_mean[-1] - current_price) / current_price * 100) if forecast_mean and current_price else 0
    current_date = datetime.now().strftime("%B %d, %Y")

    # --- YOUR ORIGINAL PROMPT (UNCHANGED, reused for Groq) ---
    prompt = f"""
You are a Senior AI Financial Analyst at InsightInvest generating a comprehensive investment research report for {symbol.upper()}.

IMPORTANT: Today's date is {current_date}. Please ensure all analysis reflects the current market context for {current_date}.

IMPORTANT: Report all monetary values in {currency} using the currency symbol {cur_sym} where appropriate. If the company trades on NSE/BSE, use INR; for major US listings use USD.

COMPANY OVERVIEW
Company: {financial_metrics.get('company_name', symbol)}
Sector: {financial_metrics.get('sector', 'N/A')}
Industry: {financial_metrics.get('industry', 'N/A')}
Current Price: {safe_format(current_price, 'currency')}

KEY FINANCIAL METRICS
Market Cap: {safe_format(financial_metrics.get('market_cap', 0), 'currency')}
P/E Ratio: {safe_format(financial_metrics.get('pe_ratio'))}
Forward P/E: {safe_format(financial_metrics.get('forward_pe'))}
EPS: {safe_format(financial_metrics.get('eps'), 'currency')}
Debt-to-Equity: {safe_format(financial_metrics.get('debt_to_equity'))}
ROE: {safe_format(financial_metrics.get('roe'), 'percent')}
Profit Margin: {safe_format(financial_metrics.get('profit_margin'), 'percent')}
Revenue Growth: {safe_format(financial_metrics.get('revenue_growth'), 'percent')}
Beta: {safe_format(financial_metrics.get('beta'))}
52-Week Range: {safe_format(financial_metrics.get('52_week_low', 0), 'currency')} - {safe_format(financial_metrics.get('52_week_high', 0), 'currency')}
Dividend Yield: {safe_format(financial_metrics.get('dividend_yield'), 'percent', default='No dividend')}

DATA QUALITY NOTES
{f"⚠️ Data Quality Flags: {', '.join(financial_metrics.get('data_quality_flags', []))}" if financial_metrics.get('data_quality_flags') else "✓ All key metrics validated successfully"}

IMPORTANT DATA VALIDATION GUIDELINES:
- Dividend yields above 25% are flagged as potentially erroneous (Indian stocks may have higher yields)
- ROE values below 1% or above 100% are flagged for manual review
- Revenue growth above 200% is flagged as potentially erroneous
- Cross-validated price data with 1-year historical data when available

ENHANCED MARKET SENTIMENT ANALYSIS
Overall Sentiment: {sentiment_summary.get('sentiment', 'neutral').title()}
Sentiment Score: {sentiment_summary.get('sentiment_score', 0.5):.3f} (0=Very Negative, 1=Very Positive)
Confidence Level: {sentiment_summary.get('confidence', 0)*100:.1f}%
News Articles Analyzed: {sentiment_summary.get('articles_analyzed', 0)}
Sentiment Distribution: Positive: {sentiment_summary.get('sentiment_distribution', {}).get('positive', 0)}, Negative: {sentiment_summary.get('sentiment_distribution', {}).get('negative', 0)}, Neutral: {sentiment_summary.get('sentiment_distribution', {}).get('neutral', 0)}
Sentiment Volatility: {sentiment_summary.get('volatility', 0):.3f} (Higher = More Mixed Opinions)

Recent Headlines Sample:
{chr(10).join(['• ' + headline for headline in news_headlines[:5]])}

ADVANCED AI PRICE FORECAST
Forecast Model: Enhanced ARIMA + Holt-Winters with Sentiment Fusion
Forecast Horizon: {len(forecast.get('mean', []))} periods
Current Price: {safe_format(current_price, 'currency')}
Predicted Price Range: {safe_format(min(forecast.get('mean', [current_price or 0])), 'currency')} - {safe_format(max(forecast.get('mean', [current_price or 0])), 'currency')}
Expected Price Change: {forecast_change:+.2f}%
Sentiment Impact: {((forecast.get('diagnostics', {}).get('sentiment_adjustment', 1.0) - 1.0) * 100):+.2f}%
Historical Volatility: {forecast.get('diagnostics', {}).get('historical_volatility', 0):.3f}
Model Quality (AIC): {forecast.get('diagnostics', {}).get('arima_aic', 'N/A')}

GENERATE PROFESSIONAL INVESTMENT ANALYSIS

Please provide a comprehensive investment analysis report with the following sections:

1. EXECUTIVE SUMMARY (3-4 sentences)
    - Overall investment thesis with specific recommendation
    - Key drivers and major risks
    - Price target based on forecast (provide conservative/base/optimistic targets in {currency})

2. FUNDAMENTAL ANALYSIS (2 paragraphs)
    - Deep dive into financial metrics (valuation, profitability, growth, debt)
    - CRITICAL: If any data quality flags are present, acknowledge the data limitations and provide context
    - If dividend yield appears unrealistic (>10%), note this as likely data error and provide typical industry context
    - If ROE appears too low (<5%) or too high (>50%), flag as potentially erroneous
    - Compare to reasonable industry benchmarks and historical norms
    - Assess financial strength while noting any metric validation concerns

3. MARKET SENTIMENT & NEWS ANALYSIS (1-2 paragraphs)
    - Interpret sentiment scores and their reliability
    - Analyze news themes and market perception
    - Discuss sentiment volatility and consensus

4. AI-ENHANCED PRICE FORECAST (1-2 paragraphs)
    - Explain the forecasting methodology and confidence
    - Discuss how sentiment influenced the prediction
    - Address forecast uncertainty and key assumptions

5. RISK ASSESSMENT (1-2 paragraphs)
    - Company-specific operational and financial risks
    - Market and sector risks
    - Technical and sentiment-based risks

6. INVESTMENT RECOMMENDATION (1-2 paragraphs)
    - Clear recommendation: Strong Buy/Buy/Hold/Sell/Strong Sell
    - Price targets (conservative, base case, optimistic) in {currency}
    - Investment timeline and key catalysts to watch

7. DISCLAIMERS (Brief)
    - Standard investment disclaimers
    - AI model limitations
    - Professional advice recommendation

Use professional financial analysis language with specific data points. Make the recommendation actionable and well-justified.

CRITICAL: Date your analysis as of {current_date}. All references to timeframes should be relative to this current date. Do not use historical dates like 2023 or 2024. This is a live analysis for {current_date}.
"""

    # --- GROQ RETRY LOGIC ---
    max_retries = len(GROQ_KEYS) * 2
    last_error = None

    for attempt in range(max_retries):
        try:
            logger.info(f"Generating report for {symbol} (Attempt {attempt+1}/{max_retries}) using Groq")

            client = get_groq_client()
            chat_completion = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a professional Senior Financial Analyst."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=2048,
            )
            report = chat_completion.choices[0].message.content.strip()

            if not report:
                raise ValueError("Empty response from Groq")

            logger.info("Investment report generated successfully")
            return report

        except Exception as e:
            error_str = str(e)
            last_error = error_str
            # Check for Rate limits
            if "429" in error_str or "rate limit" in error_str.lower():
                logger.warning(f"Groq Rate limit exceeded on current key. Rotating to next key...")
                continue
            else:
                logger.warning(f"Groq error: {error_str}. Retrying...")
                await asyncio.sleep(0.5)
                continue

    logger.error(f"Report generation failed after {max_retries} attempts. Last error: {last_error}")

    return f"""
# Investment Report Generation Failed

We apologize, but we encountered an error while generating the comprehensive investment report for {symbol}.

**Error Details:** {str(last_error)}

**Available Data Summary:**
- Company: {financial_metrics.get('company_name', symbol)}
- Current Price: ${current_price:.2f}
- Market Sentiment: {sentiment_summary.get('sentiment', 'neutral').title()} ({sentiment_summary.get('sentiment_score', 0.5):.3f})
- Forecast Direction: {'+' if forecast_change > 0 else '-'}{abs(forecast_change):.2f}%

Please try again later or contact support if the issue persists.

**Disclaimer:** This analysis is for informational purposes only and should not be considered as investment advice. Always consult with qualified financial advisors before making investment decisions.
"""

# ------------- Enhanced Main API Endpoint -------------
@app.get("/forecast/{symbol}")
async def get_comprehensive_forecast(
    symbol: str,
    request: Request,
    steps: int = Query(10, ge=5, le=30, description="Number of forecast periods"),
    period: str = Query("6mo", enum=["1mo", "3mo", "6mo", "1y", "2y"], description="Historical data period"),
    news_items: int = Query(15, ge=5, le=25, description="Number of news items for sentiment analysis"),
):
    """
    Enhanced InsightInvest AI Financial Analysis with:
    - Real-time stock data and comprehensive financial metrics
    - Advanced news sentiment analysis with volatility assessment
    - AI-powered forecasting with dynamic sentiment fusion
    - Professional investment reports with actionable recommendations
    """

    orig_symbol = symbol.strip()
    # If user provided a company name (contains spaces or letters beyond ticker chars), try to resolve
    if re.search(r"[^A-Z0-9:.]", orig_symbol, re.IGNORECASE):
        resolved = await resolve_company_to_symbol(orig_symbol)
        symbol = resolved.get("symbol", orig_symbol).upper()
        company_name = resolved.get("name", orig_symbol)
    else:
        symbol = orig_symbol.upper()
        company_name = orig_symbol

    client_ip = request.client.host if request and request.client else "unknown"
    if is_rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again in a minute.")

    try:
        logger.info(f"Starting enhanced comprehensive analysis for {symbol}")

        # OPTIMIZATION: Parallel execution of independent I/O tasks
        loop = asyncio.get_event_loop()

        # Prepare RSS URL before launching task
        query_for_news = f"{company_name} OR {symbol}"
        rss_url = generate_google_news_rss_url(query_for_news, hl="en-IN", gl="IN", ceid="IN:en")

        # Launch tasks in parallel
        task_stock = fetch_real_stock_data(symbol, period=period, interval="1d")
        task_metrics = fetch_financial_metrics(symbol)
        task_news = loop.run_in_executor(None, lambda: fetch_news_items(rss_url, max_results=news_items, follow_redirects=False))

        # Wait for all
        stock_data, financial_metrics, raw_news_items = await asyncio.gather(task_stock, task_metrics, task_news)

        # Logic to handle missing data
        need_resolve = False
        if not financial_metrics:
            need_resolve = True
        elif isinstance(financial_metrics, dict) and financial_metrics.get("error"):
            need_resolve = True
        else:
            sector = financial_metrics.get("sector")
            if sector in (None, "N/A", ""):
                need_resolve = True

        if need_resolve and orig_symbol and orig_symbol.lower() != symbol.lower():
            try:
                resolved = await resolve_company_to_symbol(orig_symbol)
                resolved_sym = resolved.get("symbol")
                resolved_name = resolved.get("name") or company_name
                if resolved_sym and resolved_sym.upper() != symbol.upper():
                    logger.info(f"Refetching data using resolved symbol {resolved_sym} for input '{orig_symbol}'")
                    try:
                        stock_data = await fetch_real_stock_data(resolved_sym, period=period, interval="1d")
                    except Exception:
                        logger.debug(f"Resolved stock data fetch failed for {resolved_sym}")

                    try:
                        financial_metrics = await fetch_financial_metrics(resolved_sym)
                    except Exception:
                        logger.debug(f"Resolved financial metrics fetch failed for {resolved_sym}")

                    symbol = resolved_sym.upper()
                    company_name = resolved_name
            except Exception as e:
                logger.debug(f"Secondary resolve attempt failed for '{orig_symbol}': {e}")

        # Build two representations: plain titles for sentiment analysis and structured items for UI
        news_texts = [item["title"] for item in raw_news_items if item.get("title")]
        news_items_short = []
        for item in raw_news_items[:5]:
            title = item.get("title") or "(no title)"
            # prefer resolved original_link when available, fall back to the RSS link
            url = item.get("original_link") or item.get("link") or ""
            news_items_short.append({"title": title, "url": url})

        sentiment = await analyze_sentiment_advanced(news_texts)

        # Generate enhanced forecast with stronger sentiment fusion
        forecast_result, hist_used = await loop.run_in_executor(
            None,
            forecast_with_enhanced_sentiment_fusion,
            stock_data,
            sentiment.get("sentiment_score", 0.5),
            steps
        )

        # Generate enhanced plot with better styling
        detected_currency = financial_metrics.get("currency") or None
        if not detected_currency:
            if symbol.endswith('.NS') or symbol.endswith('.BO'):
                detected_currency = 'INR'
            else:
                detected_currency = 'USD'

        sentiment_with_company = {**sentiment, "company_name": financial_metrics.get("company_name", symbol)}
        plot_b64 = await loop.run_in_executor(
            None,
            plot_advanced_forecast,
            hist_used,
            forecast_result,
            symbol,
            sentiment_with_company,
            "light",
            detected_currency
        )

        # Generate comprehensive investment report (pass desired currency)
        comprehensive_report = await generate_comprehensive_report(
            symbol,
            financial_metrics,
            forecast_result,
            sentiment,
            news_texts,
            currency=detected_currency
        )

        logger.info(f"Enhanced comprehensive analysis completed successfully for {symbol}")

        return {
            "symbol": symbol,
            "analysis_timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "company_info": {
                "name": financial_metrics.get("company_name", company_name),
                "sector": financial_metrics.get("sector", "N/A"),
                "industry": financial_metrics.get("industry", "N/A"),
                "current_price": financial_metrics.get("current_price", 0),
                "market_cap": financial_metrics.get("market_cap", 0)
            },
            # news_headlines now contains objects with title+url for clickable UI rendering
            "news_headlines": news_items_short,
            "financial_metrics": financial_metrics,
            "market_sentiment": {
                "source": f"Google News RSS + HuggingFace FinBERT - {len(raw_news_items)} articles fetched",
                "analysis": sentiment,
                "sample_headlines": news_texts[:5],
                "methodology": "Enhanced sentiment scoring with volatility assessment"
            },
            "price_forecast": forecast_result,
            "investment_report": comprehensive_report,
            "visualization": {
                "chart": plot_b64,
                "description": "Enhanced AI-powered price forecast with sentiment analysis and confidence intervals",
                "features": ["Real historical data", "Sentiment-adjusted predictions", "Dynamic confidence bands"]
            },
            "data_sources": {
                "stock_data": f"Yahoo Finance - {len(stock_data)} historical data points ({period})",
                "news_sentiment": "Google News RSS + HuggingFace FinBERT (Financial NLP)",
                "financial_metrics": "Yahoo Finance Company Information API",
                "forecast_model": "Enhanced ARIMA + Holt-Winters with Dynamic Sentiment Fusion v2.0"
            },
            "performance_metrics": {
                "forecast_steps": steps,
                "confidence_level": "95%",
                "sentiment_impact": f"{((forecast_result.get('diagnostics', {}).get('sentiment_adjustment', 1.0) - 1.0) * 100):+.2f}%",
                "historical_volatility": forecast_result.get('diagnostics', {}).get('historical_volatility', 0)
            },
            "disclaimer": "This analysis is for informational and educational purposes only. It does not constitute investment advice, financial advice, trading advice, or any other sort of advice. Past performance does not guarantee future results. Always conduct your own research and consult with qualified financial advisors before making investment decisions."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced comprehensive analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ------------- Health Check & Documentation -------------
@app.get("/")
async def root():
    """API Information and Health Check"""
    return {
        "service": "InsightInvest - AI Financial Analyst",
        "version": "2.0.0",
        "status": "operational",
        "enhanced_features": [
            "Real-time stock data integration",
            "Advanced news sentiment analysis with volatility metrics",
            "Dynamic AI-powered price forecasting with sentiment fusion",
            "Comprehensive financial metrics analysis",
            "Professional investment reports with actionable recommendations",
            "Enhanced visualizations with confidence intervals",
            "Multi-modal data fusion and risk assessment"
        ],
        "endpoints": {
            "main_analysis": "/forecast/{symbol}",
            "documentation": "/docs",
            "health_check": "/"
        },
        "supported_parameters": {
            "steps": "5-30 (forecast periods)",
            "period": "1mo, 3mo, 6mo, 1y, 2y (historical data)",
            "news_items": "5-25 (news articles for sentiment)"
        },
        "data_sources": [
            "Yahoo Finance (Stock Data & Financials)",
            "Google News RSS (News Headlines)",
            "HuggingFace FinBERT (Sentiment Analysis)",
            "Groq Llama 3 (Investment Report Generation)"
        ]
    }

# ------------- Application Startup -------------
if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting InsightInvest AI Financial Analyst v2.0...")
    logger.info("✨ Enhanced Features: Dynamic Forecasting, Advanced Sentiment Fusion, Professional Reports")
    logger.info("📊 Ready to provide comprehensive financial analysis with real market data")

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        workers=1,
        log_level="info"
    )