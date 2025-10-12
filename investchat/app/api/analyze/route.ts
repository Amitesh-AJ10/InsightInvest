// app/api/analyze/route.ts
import { NextRequest } from 'next/server';

// The URL of your running Python FastAPI server
const PYTHON_BACKEND_URL = 'http://127.0.0.1:8000/forecast';

export async function POST(req: NextRequest) {
  try {
    const { company } = await req.json();

    if (!company) {
      throw new Error("Company symbol is missing.");
    }

    // Make a GET request to the Python backend with the symbol in the URL
    const requestUrl = `${PYTHON_BACKEND_URL}/${company}`;
    console.log(`Forwarding request to: ${requestUrl}`);

    const pythonResponse = await fetch(requestUrl);

    if (!pythonResponse.ok) {
      const errorBody = await pythonResponse.json();
      throw new Error(errorBody.detail || 'Analysis service failed');
    }

    const data = await pythonResponse.json();
    
    // Forward the successful JSON response from Python back to the chat UI
    return new Response(JSON.stringify(data), {
      headers: { 'Content-Type': 'application/json' },
      status: 200,
    });

  } catch (error: any) {
    console.error("Error in Next.js API route:", error);
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}