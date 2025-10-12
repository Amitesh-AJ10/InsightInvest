// components/Chat.tsx
'use client';

import { useState } from 'react';
import { type CoreMessage } from 'ai';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import Image from 'next/image';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';

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
}

// A custom message type that includes our optional 'ui' property
type ExtendedMessage = CoreMessage & {
  ui?: React.ReactNode;
};

export function Chat() {
  const [messages, setMessages] = useState<ExtendedMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input || isLoading) return;

    const userMessage: ExtendedMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company: input }),
      });

      const report: AnalysisReport | { error: string } = await response.json();

      if ('error' in report || !response.ok) {
        throw new Error((report as any).error || 'Failed to get analysis.');
      }
      
      const graphDataUrl = `data:image/png;base64,${report.visualization.chart}`;

      const aiMessage: ExtendedMessage = {
        role: 'assistant',
        content: report.investment_report,
        ui: (
          <div className="mt-4 border-t border-gray-300 pt-4">
            <div className="bg-gray-50 p-3 rounded-lg">
              <h4 className="font-bold text-lg mb-2">Market Sentiment: {report.market_sentiment.analysis.sentiment.toUpperCase()}</h4>
              <p className="text-sm text-gray-700">
                Score: {report.market_sentiment.analysis.sentiment_score.toFixed(3)} | 
                Confidence: {Math.round(report.market_sentiment.analysis.confidence * 100)}%
              </p>
            </div>
            <h4 className="font-bold text-lg mt-4 mb-2">Price Forecast Chart</h4>
            <Image src={graphDataUrl} alt="Analysis Graph" width={500} height={300} className="rounded-md border shadow-sm" />
          </div>
        )
      };
      
      setMessages(prev => [...prev, aiMessage]);

    } catch (error: any) {
      const errorMessage: ExtendedMessage = { role: 'assistant', content: `Sorry, an error occurred: ${error.message}` };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="w-full max-w-3xl h-[85vh] grid grid-rows-[auto,1fr,auto] shadow-2xl">
      <CardHeader className="bg-gray-50 border-b">
        <CardTitle>InsightInvest AI Analyst 🤖</CardTitle>
        <CardDescription>Enter a stock ticker (e.g., AAPL, GOOGL) for a comprehensive AI-powered analysis.</CardDescription>
      </CardHeader>
      <CardContent className="h-full overflow-y-auto bg-gray-100 p-4">
        <ScrollArea className="h-full pr-4">
          <div className="space-y-4">
            {messages.map((m, index) => (
              <div key={index} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`p-4 rounded-lg max-w-xl shadow-md ${m.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white text-gray-800'}`}>
                  <div className="font-bold mb-2 text-lg">{m.role === 'user' ? 'You' : 'AI Analyst'}:</div>
                  <div className="prose prose-sm max-w-none">
                    <ReactMarkdown rehypePlugins={[rehypeRaw]}>
                      {m.content as string}
                    </ReactMarkdown>
                  </div>
                  {m.ui}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                  <div className="p-4 rounded-lg bg-white text-gray-600 shadow-md">
                      AI is analyzing, please wait... This may take up to 30 seconds.
                  </div>
              </div>
            )}
          </div>
        </ScrollArea>
      </CardContent>
      <div className="p-4 border-t bg-white">
        <form onSubmit={handleSubmit}>
          <div className="flex items-center gap-2">
            <Input
              value={input}
              placeholder="Enter a company ticker (e.g., MSFT)..."
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading}
            />
            <Button type="submit" disabled={isLoading}>Analyze</Button>
          </div>
        </form>
      </div>
    </Card>
  );
}