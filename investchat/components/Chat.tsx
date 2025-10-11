// components/Chat.tsx
'use client';

import { useChat } from 'ai/react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';

export function Chat() {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: '/api/analyze',
  });

  return (
    <Card className="w-full max-w-lg h-[700px] grid grid-rows-[auto,1fr,auto]">
      <CardHeader>
        <CardTitle>InvestChat AI Analyst 🤖</CardTitle>
      </CardHeader>
      <CardContent className="h-full overflow-y-auto">
        <ScrollArea className="h-full pr-4">
          {messages.map((m) => (
            <div key={m.id} className="whitespace-pre-wrap mb-4">
              <div className={`p-3 rounded-lg ${m.role === 'user' ? 'bg-blue-100' : 'bg-gray-100'}`}>
                <span className="font-bold">{m.role === 'user' ? 'You: ' : 'AI: '}</span>
                {m.content}
              </div>
            </div>
          ))}
        </ScrollArea>
      </CardContent>
      <div className="p-4 border-t">
        <form onSubmit={handleSubmit} className="flex items-center gap-2">
          <Input
            value={input}
            placeholder="Enter a company (e.g., Apple)..."
            onChange={handleInputChange}
          />
          <Button type="submit">Send</Button>
        </form>
      </div>
    </Card>
  );
}