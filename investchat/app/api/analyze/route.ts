// app/api/analyze/route.ts

import { NextRequest, NextResponse } from 'next/server';

// The URL is now dynamically pulled from environment variables.
// This is the ONLY line that needs to change.
const PYTHON_BACKEND_URL = process.env.NEXT_PUBLIC_API_URL;

export async function POST(req: NextRequest) {
  try {
    // --- No changes needed below this line ---

    if (!PYTHON_BACKEND_URL) {
      throw new Error("NEXT_PUBLIC_API_URL environment variable is not set.");
    }

    const { company } = await req.json();

    if (!company || typeof company !== 'string' || !company.trim()) {
      return NextResponse.json({ error: 'Company symbol is required.' }, { status: 400 });
    }

    // The base URL is now dynamic, pointing to either localhost or your live Render API
    const requestUrl = `${PYTHON_BACKEND_URL}/forecast/${encodeURIComponent(company.trim())}`;
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

    return NextResponse.json(data, { status: 200 });

  } catch (error: unknown) {
    console.error('Error in Next.js API route:', error);
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: message || 'Internal server error' }, { status: 500 });
  }
}