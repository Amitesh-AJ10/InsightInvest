export default function AboutPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-white to-slate-50 dark:from-gray-900 dark:to-gray-800">
      <div className="max-w-3xl p-10 bg-white rounded-2xl shadow-xl dark:bg-gray-900 border border-gray-100 dark:border-gray-700">
        <h1 className="text-3xl font-bold mb-4 text-gray-900 dark:text-gray-100">InsightInvest</h1>
        <p className="text-gray-700 dark:text-gray-300 mb-4">AI-powered investment research and forecasting. Fast news-driven sentiment, multi-market data (NSE/BSE/US), and explainable AI forecasts to help investors make informed decisions.</p>
        <ul className="list-disc pl-5 text-gray-700 dark:text-gray-300 space-y-2">
          <li>Compact financial metrics and validation flags</li>
          <li>Top news headlines with sentiment analysis</li>
          <li>AI-enhanced price forecasts with confidence bands</li>
          <li>TradingView quick links for interactive charting</li>
        </ul>
        <div className="mt-6 text-sm text-gray-500 dark:text-gray-400">Built for demo & research purposes — not financial advice.</div>
      </div>
    </div>
  )
}
