// app/api/analyze/route.ts
import { NextRequest, NextResponse } from 'next/server';

// The URL of your running Python FastAPI server
const PYTHON_BACKEND_URL = 'http://127.0.0.1:8000/forecast';

export async function POST(req: NextRequest) {
  try {
    const { company } = await req.json();

    if (!company || typeof company !== 'string' || !company.trim()) {
      return NextResponse.json({ error: 'Company symbol is required.' }, { status: 400 });
    }

    // Make a GET request to the Python backend with the symbol in the URL
    const requestUrl = `${PYTHON_BACKEND_URL}/${encodeURIComponent(company.trim())}`;
    console.log(`Forwarding request to: ${requestUrl}`);

    const pythonResponse = await fetch(requestUrl);

    const text = await pythonResponse.text();
    let data: unknown = null;
    try {
      data = JSON.parse(text);
    } catch {
      // non-JSON response from backend
      data = { message: text };
    }

    const extractMessage = (obj: unknown): string | undefined => {
      if (!obj) return undefined;
      if (typeof obj === 'string') return obj;
      if (typeof obj === 'object') {
        const o = obj as Record<string, unknown>;
        if (typeof o.detail === 'string') return o.detail;
        if (typeof o.error === 'string') return o.error;
        if (typeof o.message === 'string') return o.message;
      }
      return undefined;
    };

    if (!pythonResponse.ok) {
      const message = extractMessage(data) ?? 'Analysis service failed';
      return NextResponse.json({ error: message }, { status: pythonResponse.status || 502 });
    }

    // Forward the successful JSON response from Python back to the chat UI
    return NextResponse.json(data, { status: 200 });

  } catch (error: unknown) {
    console.error('Error in Next.js API route:', error);
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: message || 'Internal server error' }, { status: 500 });
  }
}