// components/Chat.tsx
 'use client';

import { useState, useEffect } from 'react';
import { type CoreMessage } from 'ai';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import ReactMarkdown from 'react-markdown';
import Message from './Message';

// A type that matches the rich JSON response from your Python API
interface AnalysisReport {
  investment_report: string;
  market_sentiment: {
    analysis: {
      sentiment: string;
      confidence: number;
      sentiment_score: number;
    };
  };
  visualization: {
    chart: string; // This is the base64 image string
  };
  // Optional additional fields the backend may provide
  news_headlines?: string[];
  financial_metrics?: Record<string, string | number>;
  suggestions?: {
    tradingview?: string[];
  };
  symbol?: string;
}

// A custom message type that includes our optional 'ui' property
type ExtendedMessage = CoreMessage & {
  ui?: React.ReactNode;
  timestamp?: string;
};

export function Chat() {
  const [messages, setMessages] = useState<ExtendedMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [inputError, setInputError] = useState<string | null>(null);
  // viewportRef reserved if we switch to a ref-based scroll approach later
  // Reusable analyze function so it can be used by the form submit and featured-stock clicks
  const analyzeRaw = async (rawInput: string) => {
    const raw = rawInput.trim();
    if (!raw) return;

    // Allow either a ticker (AAPL, RELIANCE.NS) OR a company/name like "Nvidia", "Nvidia Corp", "Apple Inc"
    const asUpper = raw.toUpperCase();
    const isTickerPattern = /^[A-Z]{1,12}(\.[A-Z]{1,3})?$/.test(asUpper);
    // Loose company/name pattern: letters, numbers, spaces, common punctuation, 2-80 chars
    const isNamePattern = /^[A-Za-z0-9&\-\.,'()\s]{2,80}$/.test(raw) && /[A-Za-z]/.test(raw);

    if (!isTickerPattern && !isNamePattern) {
      setInputError('Please enter a valid ticker or company name (e.g., AAPL, RELIANCE.NS, Nvidia, Nvidia Corp).');
      setIsLoading(false);
      return;
    }
    if (isLoading) return;

      // append user message inside analyzeRaw to keep a single source of truth
      const userMessage: ExtendedMessage = { role: 'user', content: raw, timestamp: new Date().toISOString() };
      setMessages(prev => [...prev, userMessage]);

    try {
      setInputError(null);
      // Send the raw user input to the backend. Backend can resolve tickers or company names via an API/Gemini call.
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company: raw }),
      });

      const report: AnalysisReport | { error: string } = await response.json();

      if ('error' in report || !response.ok) {
        const maybeErr = report && typeof report === 'object' && 'error' in report ? (report as { error?: unknown }).error : undefined;
        const errMsg = typeof maybeErr === 'string' ? maybeErr : 'Failed to get analysis.';
        throw new Error(errMsg);
      }

      const graphDataUrl = report.visualization?.chart ? `data:image/png;base64,${report.visualization.chart}` : null;

      const aiMessage: ExtendedMessage = {
        role: 'assistant',
        content: report.investment_report,
        timestamp: new Date().toISOString(),
        ui: (
          <div className="space-y-6">
            {/* Market Sentiment Section */}
            <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-slate-50 to-white border border-gray-200/50 p-6 shadow-sm">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0">
                  <div className={`h-12 w-12 rounded-full flex items-center justify-center ${
                    report.market_sentiment.analysis.sentiment.toLowerCase() === 'positive' 
                      ? 'bg-green-100 text-green-700' 
                      : report.market_sentiment.analysis.sentiment.toLowerCase() === 'negative'
                      ? 'bg-red-100 text-red-700'
                      : 'bg-yellow-100 text-yellow-700'
                  }`}>
                    {report.market_sentiment.analysis.sentiment.toLowerCase() === 'positive' ? (
                      <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                      </svg>
                    ) : report.market_sentiment.analysis.sentiment.toLowerCase() === 'negative' ? (
                      <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
                      </svg>
                    ) : (
                      <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                      </svg>
                    )}
                  </div>
                </div>
                <div className="flex-1">
                  <h4 className="text-lg font-bold text-gray-900 mb-2">
                    Market Sentiment: <span className={`${
                      report.market_sentiment.analysis.sentiment.toLowerCase() === 'positive' 
                        ? 'text-green-600' 
                        : report.market_sentiment.analysis.sentiment.toLowerCase() === 'negative'
                        ? 'text-red-600'
                        : 'text-yellow-600'
                    }`}>
                      {report.market_sentiment.analysis.sentiment.toUpperCase()}
                    </span>
                  </h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-white/70 rounded-lg p-3 border border-gray-100">
                      <div className="text-gray-600 mb-1">Sentiment Score</div>
                      <div className="font-semibold text-gray-900">{report.market_sentiment.analysis.sentiment_score.toFixed(3)}</div>
                    </div>
                    <div className="bg-white/70 rounded-lg p-3 border border-gray-100">
                      <div className="text-gray-600 mb-1">Confidence Level</div>
                      <div className="font-semibold text-gray-900">{Math.round(report.market_sentiment.analysis.confidence * 100)}%</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Price Forecast Chart Section */}
            <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-slate-50 to-white border border-gray-200/50 p-6 shadow-sm">
              <div className="flex items-center gap-3 mb-4">
                <div className="h-10 w-10 rounded-lg bg-blue-100 flex items-center justify-center">
                  <svg className="h-5 w-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <h4 className="text-lg font-bold text-gray-900">AI Price Forecast</h4>
              </div>
              {graphDataUrl ? (
                <div className="relative rounded-lg overflow-hidden bg-white border border-gray-200/50 shadow-sm">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img 
                    src={graphDataUrl} 
                    alt="AI-generated price forecast chart showing predicted stock price movements" 
                    className="w-full h-auto"
                  />
                  <div className="absolute top-3 right-3 bg-white/90 backdrop-blur-sm rounded-lg px-3 py-1 text-xs font-medium text-gray-700 border border-gray-200/50">
                    AI Generated
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center h-48 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
                  <div className="text-center">
                    <svg className="h-12 w-12 text-gray-400 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    <div className="text-sm text-gray-500">No chart available for this analysis</div>
                  </div>
                </div>
              )}
            </div>
            {/* Top News Headlines (3-5) */}
            {report.news_headlines && report.news_headlines.length > 0 && (
              <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-slate-50 to-white border border-gray-200/50 p-4 shadow-sm">
                <h4 className="text-lg font-bold text-gray-900 mb-3">Top News</h4>
                <ul className="list-disc list-inside space-y-2 text-sm text-gray-700">
                  {report.news_headlines.slice(0, 5).map((item: any, i: number) => (
                    <li key={i} className="truncate">
                      {item && item.url ? (
                        <a href={item.url} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">
                          {item.title}
                        </a>
                      ) : (
                        <span>{item.title || String(item)}</span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Key Financial Metrics (compact) */}
            {report.financial_metrics && Object.keys(report.financial_metrics).length > 0 && (
              <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-slate-50 to-white border border-gray-200/50 p-4 shadow-sm">
                <h4 className="text-lg font-bold text-gray-900 mb-3">Key Metrics</h4>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
                  {/* Preferred keys: company_name, sector, industry, market_cap, pe_ratio, forward_pe, eps, debt_to_equity, revenue_trend, profit_margin_trend */}
                  {(() => {
                    const fm = report.financial_metrics || {};
                    const rows: Array<[string, string]> = [];

                    // helper: compact number formatting (e.g., 94.04B)
                    const compactNumber = (v: any) => {
                      const n = Number(v);
                      if (!isFinite(n)) return String(v);
                      const abs = Math.abs(n);
                      if (abs >= 1e12) return (n / 1e12).toFixed(2) + 'T';
                      if (abs >= 1e9) return (n / 1e9).toFixed(2) + 'B';
                      if (abs >= 1e6) return (n / 1e6).toFixed(2) + 'M';
                      return n.toLocaleString();
                    };

                    // helper: format array of numbers to human-friendly string
                    const formatNumberArray = (arr: any[]) => {
                      try {
                        return arr.map(x => compactNumber(x)).join(', ');
                      } catch {
                        return String(arr);
                      }
                    };

                    // helper: format percent array
                    const formatPercentArray = (arr: any[]) => {
                      try {
                        return arr.map(x => {
                          const n = Number(x);
                          if (!isFinite(n)) return String(x);
                          return `${n.toFixed(2)}%`;
                        }).join(', ');
                      } catch {
                        return String(arr);
                      }
                    };

                    // Prioritize the requested metrics: EPS, D/E, quarterly revenue and profit margin trends
                    const fmtFloat = (v: any, d = 2) => {
                      const n = Number(v);
                      if (!isFinite(n)) return 'N/A';
                      return n.toFixed(d);
                    };

                    if (fm['eps']) rows.push(['EPS', fmtFloat(fm['eps'])]);
                    if (fm['debt_to_equity'] || fm['debt_equity']) rows.push(['D/E Ratio', fmtFloat(fm['debt_to_equity'] || fm['debt_equity'])]);
                    // Prioritize P/E so it appears near top
                    if (fm['pe_ratio'] || fm['pe']) rows.push(['P/E Ratio', fmtFloat(fm['pe_ratio'] || fm['pe'])]);
                    if (fm['quarterly_revenue_trend'] && Array.isArray(fm['quarterly_revenue_trend']) && fm['quarterly_revenue_trend'].length) rows.push(['Quarterly Revenue', formatNumberArray(fm['quarterly_revenue_trend'])]);
                    if (fm['quarterly_profit_margin_trend'] && Array.isArray(fm['quarterly_profit_margin_trend']) && fm['quarterly_profit_margin_trend'].length) rows.push(['Quarterly Profit Margin', formatPercentArray(fm['quarterly_profit_margin_trend'])]);

                    if (fm['company_name']) rows.push(['Company', String(fm['company_name'])]);
                    if (fm['sector']) rows.push(['Sector', String(fm['sector'])]);
                    if (fm['industry']) rows.push(['Industry', String(fm['industry'])]);
                    if (fm['market_cap']) {
                      // human friendly fmt
                      const n = Number(fm['market_cap']) || 0;
                      const fmt = n >= 1e12 ? (n/1e12).toFixed(2)+'T' : n >= 1e9 ? (n/1e9).toFixed(2)+'B' : n >= 1e6 ? (n/1e6).toFixed(2)+'M' : n.toLocaleString();
                      rows.push(['Market Cap', fmt]);
                    }
                    if (fm['forward_pe']) rows.push(['Forward P/E', String(fm['forward_pe'])]);
                    // keep EPS/D/E already added above; avoid duplicates
                    if (fm['revenue_trend']) rows.push(['Revenue Trend', String(fm['revenue_trend'])]);
                    if (fm['profit_margin_trend']) rows.push(['Profit Margin', String(fm['profit_margin_trend'])]);
                    // fallback: show first 6 entries if nothing matched
                    if (rows.length === 0) {
                      return Object.entries(fm).slice(0,6).map(([k,v]) => (
                        <div key={k} className="bg-white/70 rounded-lg p-3 border border-gray-100">
                          <div className="text-gray-600 mb-1 truncate">{k.replace(/_/g,' ')}</div>
                          <div className="font-semibold text-gray-900">{String(v)}</div>
                        </div>
                      ));
                    }
                    return rows.slice(0,6).map(([label, val]) => (
                      <div key={label} className="bg-white/70 rounded-lg p-3 border border-gray-100">
                        <div className="text-gray-600 mb-1 truncate">{label}</div>
                        <div className="font-semibold text-gray-900">{val}</div>
                      </div>
                    ));
                  })()}
                </div>
              </div>
            )}

            {/* TradingView suggestions (static fallback) */}
            <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-slate-50 to-white border border-gray-200/50 p-4 shadow-sm">
              <h4 className="text-lg font-bold text-gray-900 mb-2">TradingView Suggestions</h4>
              <div className="text-sm text-gray-700 mb-3">Indicators & timeframes you can try on TradingView</div>
              <div className="flex flex-wrap gap-2">
                {(report.suggestions?.tradingview || [
                  'EMA(20), EMA(50)',
                  'RSI(14) - look for divergence/overbought',
                  'MACD(12,26,9) - crossovers',
                  'Ichimoku Cloud - trend confirmation',
                  'Timeframes: 1D, 4H, 1W'
                ]).map((s, idx) => (
                  <div key={idx} className="px-3 py-1 bg-white/80 border border-gray-100 rounded-full text-xs font-medium text-gray-800">{s}</div>
                ))}
              </div>
              {/* Quick TradingView link */}
              <div className="mt-4 flex items-center gap-3">
                <a
                  target="_blank"
                  rel="noreferrer"
                  href={report.symbol ? `https://www.tradingview.com/symbols/${encodeURIComponent(report.symbol)}/` : 'https://www.tradingview.com/' }
                  className="inline-flex items-center gap-2 px-3 py-2 bg-blue-600 text-white rounded-lg text-sm hover:opacity-95"
                >
                  Open interactive chart
                </a>
                <div className="text-xs text-gray-500">(opens TradingView search for the symbol)</div>
              </div>
            </div>

            {/* Metric definitions */}
            <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-white to-white/80 border border-gray-100 p-4 shadow-sm">
              <h4 className="text-sm font-bold text-gray-900 mb-2">Metric definitions</h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-gray-700">
                <div><strong>P/E Ratio</strong>: Price per share ÷ Earnings per share (higher = more expensive).</div>
                <div><strong>EPS</strong>: Earnings per share — company's profit allocated per share.</div>
                <div><strong>D/E Ratio</strong>: Debt-to-Equity — company's leverage level.</div>
                <div><strong>Revenue / Profit Margin</strong>: Trends show recent quarterly revenue and margin performance.</div>
              </div>
            </div>
          </div>
        )
      };
      
  setMessages(prev => [...prev, aiMessage]);

    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      const errorMessage: ExtendedMessage = { role: 'assistant', content: `Sorry, an error occurred: ${message}`, timestamp: new Date().toISOString() };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Form submit wrapper that uses analyzeRaw
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const raw = input.trim();
    if (!raw) return;
    setIsLoading(true);
    await analyzeRaw(raw);
    setInput('');
  };

  // Auto-scroll effect: scroll to bottom whenever messages change
  useEffect(() => {
    // best-effort: find the chat scroll container inside this component
    const scrollable = document.querySelector('.h-full.pr-4') as HTMLElement | null;
    if (scrollable) {
      // small timeout to wait for DOM updates
      setTimeout(() => {
        scrollable.scrollTop = scrollable.scrollHeight;
      }, 50);
    }
  }, [messages, isLoading]);

  // Click listener: detect clicks on featured-stock tiles and trigger analysis
    useEffect(() => {
    const handler = (ev: MouseEvent) => {
      const target = ev.target as HTMLElement | null;
      if (!target) return;
      // Find the closest ancestor whose class contains 'group/item' or 'group-item'
      const el = target.closest('[class*="group/item"], [class*="group-item"], .group-item') as HTMLElement | null;
      if (!el) return;

      // Prefer the .font-semibold element (ticker text), else first div
      const symbolNode = el.querySelector('.font-semibold') || el.querySelector('div');
      if (!symbolNode) return;
      const rawText = (symbolNode.textContent || '').trim();
      // Extract first token that looks like a ticker (letters, numbers, dot, dash)
      const m = rawText.match(/[A-Z0-9\.\-]{1,20}/i);
      if (!m) return;
      const ticker = m[0].toUpperCase();

      // If already loading or input matches current ticker, do nothing
      if (isLoading) return;
      if (input.trim().toUpperCase() === ticker) return;

      // Populate input then call analyzeRaw once (analyzeRaw appends the user message)
      setInput(ticker);
      setIsLoading(true);
      setTimeout(async () => {
        await analyzeRaw(ticker);
      }, 50);
    };

    document.addEventListener('click', handler);
    return () => document.removeEventListener('click', handler);
  }, [input, isLoading]);

  return (
    <div className="relative">
      <Card className="w-full h-[85vh] grid grid-rows-[auto,1fr,auto] backdrop-blur-sm bg-card border-border shadow-2xl rounded-2xl overflow-hidden">
        <CardHeader className="bg-gradient-to-r from-white/90 to-white/70 backdrop-blur-sm border-b border-white/30 p-6">
          <div className="flex items-center gap-4">
            <div className="relative">
              <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-lg">
                <svg className="h-6 w-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <div className="absolute -bottom-1 -right-1 h-4 w-4 rounded-full bg-green-500 border-2 border-white flex items-center justify-center">
                <div className="h-2 w-2 rounded-full bg-white animate-pulse"></div>
              </div>
            </div>
            <div>
              <CardTitle className="text-xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">InsightInvest AI Analyst</CardTitle>
              <CardDescription className="text-gray-600">Advanced financial analysis powered by artificial intelligence</CardDescription>
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="h-full overflow-hidden bg-gradient-to-b from-slate-50/50 to-white/50 p-0">
          <ScrollArea className="h-full">
            <div className="p-6 space-y-6">
              {messages.length === 0 && (
                <div className="flex items-center justify-center h-full py-20">
                  <div className="text-center max-w-md">
                    <div className="h-16 w-16 rounded-full bg-gradient-to-br from-blue-100 to-indigo-100 flex items-center justify-center mx-auto mb-4">
                      <svg className="h-8 w-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-semibold text-gray-800 mb-2">Ready to analyze your investments</h3>
                    <p className="text-gray-600 text-sm">Enter a stock ticker below to get comprehensive AI-powered financial analysis, market sentiment, and price forecasts.</p>
                  </div>
                </div>
              )}
              
              {messages.map((m, index) => (
                <div key={`${m.role}-${index}-${String(m.content).slice(0,12)}`} className="animate-fade-in">
                  <Message role={m.role} content={m.content as string} ui={m.ui} timestamp={m.timestamp} />
                </div>
              ))}
              
              {isLoading && (
                <div className="animate-fade-in">
                  <div className="flex items-start gap-3">
                    <div className="flex-shrink-0 h-10 w-10 rounded-full bg-gradient-to-br from-blue-100 to-indigo-100 flex items-center justify-center">
                      <svg className="h-5 w-5 text-blue-600 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
                      </svg>
                    </div>
                    <div className="rounded-2xl bg-white/80 backdrop-blur-sm border border-gray-200/50 p-4 shadow-lg max-w-xs">
                      <div className="flex items-center gap-3">
                        <div className="flex space-x-1">
                          <div className="h-2 w-2 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                          <div className="h-2 w-2 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                          <div className="h-2 w-2 bg-blue-500 rounded-full animate-bounce"></div>
                        </div>
                        <span className="text-sm text-gray-700 font-medium">AI is analyzing your request...</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
        
        <div className="bg-gradient-to-r from-white/95 to-white/90 backdrop-blur-sm border-t border-white/30 p-6">
          <form onSubmit={handleSubmit} aria-label="Analyze company form">
            <div className="flex items-center gap-3">
              <div className="relative flex-1">
                <Input
                  value={input}
                  placeholder="Enter stock ticker (e.g., AAPL, MSFT, GOOGL)..."
                  onChange={(e) => setInput(e.target.value)}
                  disabled={isLoading}
                  className="pl-12 pr-4 py-3 text-base rounded-xl border-border bg-background/80 backdrop-blur-sm focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 transition-all duration-200 placeholder:text-muted-foreground"
                />
                <div className="absolute left-4 top-1/2 -translate-y-1/2">
                  <svg className="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </div>
              </div>
              <Button 
                type="submit" 
                disabled={isLoading || !input.trim()} 
                className="px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isLoading ? (
                  <>
                    <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
                    </svg>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Analyze
                  </>
                )}
              </Button>
            </div>
            {inputError && (
              <div className="mt-3 p-3 rounded-lg bg-red-50 border border-red-200 flex items-center gap-2">
                <svg className="h-4 w-4 text-red-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-sm text-red-700">{inputError}</span>
              </div>
            )}
          </form>
        </div>
      </Card>
    </div>
  );
}
