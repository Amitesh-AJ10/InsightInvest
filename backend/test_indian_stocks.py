#!/usr/bin/env python3
"""
Test script to verify Indian stock market support in InsightInvest
Tests both NSE (.NS) and BSE (.BO) stocks
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from main import normalize_stock_symbol, fetch_real_stock_data

async def test_indian_stocks():
    """Test Indian stock data fetching capabilities"""
    print("🇮🇳 Testing Indian Stock Market Support")
    print("=" * 50)
    
    # Test stocks
    test_symbols = [
        "RELIANCE",     # Auto-detect NSE
        "RELIANCE.NS",  # Explicit NSE
        "RELIANCE.BO",  # Explicit BSE
        "TCS",          # Auto-detect NSE
        "TCS.NS",       # Explicit NSE
        "INFY.NS",      # Infosys NSE
        "ICICIBANK.NS", # ICICI Bank NSE
        "SBIN.NS",      # State Bank of India NSE
    ]
    
    print(f"Testing {len(test_symbols)} Indian stock symbols...\n")
    
    for symbol in test_symbols:
        try:
            print(f"📊 Testing: {symbol}")
            
            # Test symbol normalization
            normalized = normalize_stock_symbol(symbol)
            print(f"   Normalized: {symbol} → {normalized}")
            
            # Test data fetching (small period for quick test)
            data = await fetch_real_stock_data(symbol, period="5d", interval="1d")
            
            if not data.empty:
                latest_price = data['Close'].iloc[-1]
                print(f"   ✅ Success: Latest price ₹{latest_price:.2f}")
                print(f"   📈 Data points: {len(data)} days")
            else:
                print("   ❌ No data available")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        print()
    
    print("🌍 Testing International Markets")
    print("=" * 50)
    
    international_symbols = [
        "AAPL",      # US stock
        "MSFT",      # US stock  
        "GOOGL",     # US stock
        "TSLA.L",    # Tesla London (if available)
        "7203.T",    # Toyota Japan
    ]
    
    for symbol in international_symbols:
        try:
            print(f"📊 Testing: {symbol}")
            normalized = normalize_stock_symbol(symbol)
            print(f"   Normalized: {symbol} → {normalized}")
            
            data = await fetch_real_stock_data(symbol, period="5d", interval="1d")
            
            if not data.empty:
                latest_price = data['Close'].iloc[-1]
                print(f"   ✅ Success: Latest price ${latest_price:.2f}")
                print(f"   📈 Data points: {len(data)} days")
            else:
                print("   ❌ No data available")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        print()

if __name__ == "__main__":
    print("🚀 InsightInvest Multi-Market Stock Data Test")
    print("Testing support for Indian and international markets")
    print()
    
    asyncio.run(test_indian_stocks())
    
    print("✅ Test completed!")
    print("\n💡 Usage tips:")
    print("   • For Indian stocks: Use RELIANCE.NS (NSE) or RELIANCE.BO (BSE)")
    print("   • Popular symbols auto-detect NSE: RELIANCE → RELIANCE.NS")
    print("   • For US stocks: Use AAPL, MSFT, GOOGL (no suffix needed)")
    print("   • For UK stocks: Use SYMBOL.L (London Stock Exchange)")