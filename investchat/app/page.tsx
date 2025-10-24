// app/page.tsx
import { Chat } from '@/components/Chat'
import Header from '@/components/Header'
import Footer from '@/components/Footer'

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <main className="flex-grow mx-auto w-full max-w-7xl px-4 sm:px-6 py-4 sm:py-8">
        <div className="flex flex-col lg:grid lg:grid-cols-4 gap-4 lg:gap-8 items-start">
          <section className="w-full lg:col-span-1 order-2 lg:order-1">
            <div className="lg:sticky lg:top-8">
              <div className="group relative overflow-hidden rounded-xl lg:rounded-2xl bg-card backdrop-blur-sm border border-border p-4 sm:p-6 lg:p-8 shadow-lg lg:shadow-xl hover:shadow-xl lg:hover:shadow-2xl transition-all duration-300 hover:-translate-y-1">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-indigo-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <div className="relative z-10">
                  <div className="flex items-center gap-2 sm:gap-3 mb-3 sm:mb-4">
                    <div className="h-10 w-10 sm:h-12 sm:w-12 rounded-lg sm:rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-lg">
                      <svg className="h-5 w-5 sm:h-6 sm:w-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    </div>
                    <h1 className="text-lg sm:text-xl lg:text-2xl font-bold text-foreground">AI Investment Analyst</h1>
                  </div>
                  <p className="text-sm sm:text-base text-muted-foreground leading-relaxed mb-4 sm:mb-6">Get comprehensive investment reports, market sentiment analysis, and AI-powered price forecasts for any stock ticker.</p>

                  <div className="space-y-4">
                    <div>
                      <h3 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full bg-blue-500"></span>
                        Featured Stocks
                      </h3>
                      <div className="space-y-3">
                        {/* US Stocks */}
                        <div className="group/item flex items-center justify-between p-3 rounded-lg bg-gradient-to-r from-gray-50 to-gray-100/50 border border-gray-200/50 hover:border-blue-200 hover:bg-gradient-to-r hover:from-blue-50/50 hover:to-indigo-50/30 transition-all duration-200 cursor-pointer">
                          <div>
                            <div className="font-semibold text-gray-800">AAPL</div>
                            <div className="text-xs text-gray-600">Apple Inc. (US)</div>
                          </div>
                          <svg className="h-4 w-4 text-gray-400 group-hover/item:text-blue-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </div>
                        <div className="group/item flex items-center justify-between p-3 rounded-lg bg-gradient-to-r from-gray-50 to-gray-100/50 border border-gray-200/50 hover:border-blue-200 hover:bg-gradient-to-r hover:from-blue-50/50 hover:to-indigo-50/30 transition-all duration-200 cursor-pointer">
                          <div>
                            <div className="font-semibold text-gray-800">MSFT</div>
                            <div className="text-xs text-gray-600">Microsoft Corp. (US)</div>
                          </div>
                          <svg className="h-4 w-4 text-gray-400 group-hover/item:text-blue-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </div>
                        
                        {/* Indian Stocks */}
                        <div className="group/item flex items-center justify-between p-3 rounded-lg bg-gradient-to-r from-orange-50 to-green-50 border border-orange-200/50 hover:border-orange-300 hover:bg-gradient-to-r hover:from-orange-100/50 hover:to-green-100/30 transition-all duration-200 cursor-pointer">
                          <div>
                            <div className="font-semibold text-gray-800 flex items-center gap-2">
                              RELIANCE.NS
                              <span className="text-xs px-2 py-0.5 bg-orange-100 text-orange-700 rounded-full font-medium">🇮🇳 NSE</span>
                            </div>
                            <div className="text-xs text-gray-600">Reliance Industries</div>
                          </div>
                          <svg className="h-4 w-4 text-gray-400 group-hover/item:text-orange-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </div>
                        
                        <div className="group/item flex items-center justify-between p-3 rounded-lg bg-gradient-to-r from-orange-50 to-green-50 border border-orange-200/50 hover:border-orange-300 hover:bg-gradient-to-r hover:from-orange-100/50 hover:to-green-100/30 transition-all duration-200 cursor-pointer">
                          <div>
                            <div className="font-semibold text-gray-800 flex items-center gap-2">
                              TCS.NS
                              <span className="text-xs px-2 py-0.5 bg-orange-100 text-orange-700 rounded-full font-medium">🇮🇳 NSE</span>
                            </div>
                            <div className="text-xs text-gray-600">Tata Consultancy Services</div>
                          </div>
                          <svg className="h-4 w-4 text-gray-400 group-hover/item:text-orange-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </div>
                        <div className="group/item flex items-center justify-between p-3 rounded-lg bg-gradient-to-r from-gray-50 to-gray-100/50 border border-gray-200/50 hover:border-blue-200 hover:bg-gradient-to-r hover:from-blue-50/50 hover:to-indigo-50/30 transition-all duration-200 cursor-pointer">
                          <div>
                            <div className="font-semibold text-gray-800">GOOGL</div>
                            <div className="text-xs text-gray-600">Alphabet Inc. (US)</div>
                          </div>
                          <svg className="h-4 w-4 text-gray-400 group-hover/item:text-blue-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </div>
                      </div>
                    </div>

                    {/* Supported Markets Info */}
                    <div className="mt-6 pt-6 border-t border-gray-200/50">
                      <h3 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full bg-green-500"></span>
                        Supported Markets
                      </h3>
                      <div className="space-y-2 text-xs text-gray-600">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-gray-700">🇺🇸 US:</span>
                          <span>AAPL, MSFT, GOOGL</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-gray-700">🇮🇳 India:</span>
                          <span>RELIANCE.NS, TCS.NS</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-gray-700">🌍 Global:</span>
                          <span>SYMBOL.L (UK), .T (Japan)</span>
                        </div>
                        <div className="mt-3 p-3 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200/30">
                          <p className="text-xs text-blue-700">
                            <strong>💡 Pro Tip:</strong> Add .NS for NSE or .BO for BSE when analyzing Indian stocks
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="pt-4 border-t border-gray-200/50">
                      <div className="flex items-center gap-2 text-xs text-gray-500">
                        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        AI-powered analysis • Real-time data • Multi-market support
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
          <section className="w-full lg:col-span-3 order-1 lg:order-2">
            <Chat />
          </section>
        </div>
      </main>
      <Footer />
    </div>
  )
}