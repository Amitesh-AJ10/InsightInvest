// app/page.tsx
import { Chat } from "@/components/Chat";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-200 p-4">
      <Chat />
    </main>
  );
}