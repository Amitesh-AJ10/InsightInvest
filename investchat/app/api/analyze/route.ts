// app/api/analyze/route.ts

import { NextRequest, NextResponse } from 'next/server';

// Ensure this is set in your .env.local file:
// NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
const PYTHON_BACKEND_URL = process.env.NEXT_PUBLIC_API_URL;

export async function POST(req: NextRequest) {
  try {
    if (!PYTHON_BACKEND_URL) {
      throw new Error("NEXT_PUBLIC_API_URL environment variable is not set.");
    }

    const { company } = await req.json();

    if (!company || typeof company !== 'string' || !company.trim()) {
      return NextResponse.json({ error: 'Company symbol is required.' }, { status: 400 });
    }

    // Construct the URL
    const requestUrl = `${PYTHON_BACKEND_URL}/forecast/${encodeURIComponent(company.trim())}`;
    console.log(`Forwarding request to: ${requestUrl}`);

    // --- FIX: Add AbortSignal for longer timeout & disable caching ---
    // Create a controller to allow 60 second timeout (instead of default 10s)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 seconds

    try {
      const pythonResponse = await fetch(requestUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        cache: 'no-store', // Disable Next.js caching
        signal: controller.signal, // Apply the 60s timeout
        // @ts-ignore - Next.js specific extension
        next: { revalidate: 0 }
      });

      clearTimeout(timeoutId); // Clear timeout if successful

      const text = await pythonResponse.text();
      let data: any = null;

      try {
        data = JSON.parse(text);
      } catch {
        data = { message: text };
      }

      if (!pythonResponse.ok) {
        const message = data.detail || data.error || 'Analysis service failed';
        return NextResponse.json({ error: message }, { status: pythonResponse.status || 502 });
      }

      return NextResponse.json(data, { status: 200 });

    } catch (fetchError: any) {
      clearTimeout(timeoutId);
      if (fetchError.name === 'AbortError') {
        throw new Error('Analysis timed out. The backend took too long to respond.');
      }
      throw fetchError;
    }

  } catch (error: any) {
    console.error('Error in Next.js API route:', error);
    // Improve error message for connection refused
    let message = error.message || String(error);
    if (message.includes('ECONNREFUSED')) {
      message = "Cannot connect to Backend. Is 'uvicorn' running on port 8000?";
    }
    return NextResponse.json({ error: message }, { status: 500 });
  }
}