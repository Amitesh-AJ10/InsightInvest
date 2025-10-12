#!/usr/bin/env python3
"""
Test the improved financial metrics validation for Visa (V) and other stocks
to ensure data quality issues are properly handled.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from main import fetch_financial_metrics

async def test_metrics_validation():
    """Test the new metrics validation system"""
    print("🔍 Testing Enhanced Financial Metrics Validation")
    print("=" * 60)
    
    test_symbols = [
        "V",        # Visa - previously had issues
        "AAPL",     # Apple - typically clean data
        "MSFT",     # Microsoft - typically clean data  
        "TCS.NS",   # Indian stock - Tata Consultancy Services
    ]
    
    for symbol in test_symbols:
        print(f"\n📊 Testing Financial Metrics for: {symbol}")
        print("-" * 40)
        
        try:
            metrics = await fetch_financial_metrics(symbol)
            
            # Display key metrics with validation status
            print(f"Company: {metrics.get('company_name', 'N/A')}")
            print(f"Sector: {metrics.get('sector', 'N/A')}")
            
            # Check dividend yield
            div_yield = metrics.get('dividend_yield')
            if div_yield is not None:
                print(f"✓ Dividend Yield: {div_yield:.2f}%")
                if div_yield > 10:
                    print("  ⚠️  WARNING: Dividend yield above 10% - likely data error")
            else:
                print("❌ Dividend Yield: Data validation failed")
            
            # Check ROE
            roe = metrics.get('roe')
            if roe is not None:
                print(f"✓ ROE: {roe:.2f}%")
                if roe < 1:
                    print("  ⚠️  WARNING: ROE below 1% - unusually low")
                elif roe > 50:
                    print("  ⚠️  WARNING: ROE above 50% - unusually high")
            else:
                print("❌ ROE: Data validation failed")
            
            # Check profit margin
            profit_margin = metrics.get('profit_margin')
            if profit_margin is not None:
                print(f"✓ Profit Margin: {profit_margin:.2f}%")
            else:
                print("❌ Profit Margin: Data validation failed")
            
            # Check revenue growth
            revenue_growth = metrics.get('revenue_growth')
            if revenue_growth is not None:
                print(f"✓ Revenue Growth: {revenue_growth:.2f}%")
                if abs(revenue_growth) > 100:
                    print("  ⚠️  WARNING: Revenue growth >100% - verify accuracy")
            else:
                print("❌ Revenue Growth: Data validation failed")
            
            # Check PE ratios
            pe = metrics.get('pe_ratio')
            forward_pe = metrics.get('forward_pe')
            print(f"P/E Ratio: {pe if pe else 'N/A'}")
            print(f"Forward P/E: {forward_pe if forward_pe else 'N/A'}")
            
            # Display data quality flags
            quality_flags = metrics.get('data_quality_flags', [])
            if quality_flags:
                print("\n🚨 Data Quality Issues Found:")
                for flag in quality_flags:
                    print(f"  • {flag}")
            else:
                print("\n✅ No data quality issues detected")
                
        except Exception as e:
            print(f"❌ Error fetching metrics for {symbol}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Financial Metrics Validation Test Completed!")

if __name__ == "__main__":
    print("🚀 InsightInvest Enhanced Metrics Validation Test")
    print("Testing improved data validation for financial metrics")
    print()
    
    asyncio.run(test_metrics_validation())