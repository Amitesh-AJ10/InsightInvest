// app/api/analyze/route.ts
import { streamText } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { NextRequest } from 'next/server';

export const runtime = 'edge';

// Initialize the OpenAI client (it will read the API key from .env.local)
const openai = createOpenAI();

export async function POST(req: NextRequest) {
  const { messages } = await req.json();

  // Call the AI with the user's messages
  const result = await streamText({
    model: openai('gpt-4-turbo'),
    messages,
    system: 'You are a helpful financial analyst chatbot.',
  });

  // Respond with the stream using the new, recommended method
  return result.response;
}