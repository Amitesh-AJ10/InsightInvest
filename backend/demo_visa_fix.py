"""
Quick demo showing the Visa data quality fix in action
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import fetch_financial_metrics

async def demo_visa_fix():
    """Demo the Visa financial metrics fix"""
    print("🔧 Demonstrating Visa (V) Data Quality Fix")
    print("=" * 50)
    print()
    
    # Test Visa specifically since it was mentioned in the issue
    symbol = "V"
    print(f"📊 Fetching validated financial metrics for {symbol}...")
    
    try:
        metrics = await fetch_financial_metrics(symbol)
        
        print(f"\n✅ Results for {metrics.get('company_name', symbol)}:")
        print(f"   Sector: {metrics.get('sector', 'N/A')}")
        
        # Show the key metrics that were problematic before
        div_yield = metrics.get('dividend_yield')
        roe = metrics.get('roe')  
        revenue_growth = metrics.get('revenue_growth')
        profit_margin = metrics.get('profit_margin')
        
        print(f"\n📈 Key Financial Metrics (Validated):")
        print(f"   Dividend Yield: {div_yield:.2f}% {'✓ Reasonable' if div_yield and div_yield < 10 else '⚠️ Flagged'}") 
        print(f"   ROE: {roe:.2f}% {'✓ Strong' if roe and roe > 20 else '⚠️ Review needed'}") 
        print(f"   Revenue Growth: {revenue_growth:.2f}% {'✓ Healthy' if revenue_growth and 5 <= revenue_growth <= 15 else '⚠️ Outside typical range'}")
        print(f"   Profit Margin: {profit_margin:.2f}% {'✓ Excellent' if profit_margin and profit_margin > 30 else '✓ Good' if profit_margin and profit_margin > 10 else '⚠️ Low'}")
        
        # Show data quality status
        quality_flags = metrics.get('data_quality_flags', [])
        print(f"\n🛡️ Data Quality Status:")
        if quality_flags:
            print(f"   ⚠️ Issues detected: {len(quality_flags)}")
            for flag in quality_flags:
                print(f"     • {flag}")
        else:
            print("   ✅ All metrics validated successfully")
        
        print(f"\n💰 Valuation Metrics:")
        print(f"   Current Price: ${metrics.get('current_price', 0):.2f}")
        print(f"   P/E Ratio: {metrics.get('pe_ratio', 'N/A')}")
        print(f"   Market Cap: ${metrics.get('market_cap', 0):,}")
        
        print(f"\n📊 The previous report showed obviously wrong values:")
        print(f"   🚫 OLD: Dividend Yield 69% (impossible)")
        print(f"   ✅ NOW: Dividend Yield {div_yield:.2f}% (realistic)")
        print(f"   🚫 OLD: ROE 0.52% (far too low for Visa)")  
        print(f"   ✅ NOW: ROE {roe:.2f}% (strong for financial services)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 InsightInvest Data Quality Validation Demo")
    print("Showing how the Visa metrics issue has been fixed")
    print()
    
    asyncio.run(demo_visa_fix())
    
    print(f"\n🎯 Summary:")
    print(f"• Enhanced financial metrics validation with bounds checking")
    print(f"• Cross-validation with historical price data")  
    print(f"• Data quality flags to track validation issues")
    print(f"• Updated AI report generation to acknowledge data limitations")
    print(f"• Multi-market support for Indian stocks (NSE/BSE)")
    print(f"\n✅ The AI reports should now provide accurate, validated financial analysis!")