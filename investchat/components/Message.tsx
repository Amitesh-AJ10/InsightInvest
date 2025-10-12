"use client"

import React from 'react'
import { ClipboardCopy } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

type Props = {
  role: 'user' | 'assistant' | string
  content: string
  ui?: React.ReactNode
  timestamp?: string
}

export default function Message({ role, content, ui, timestamp }: Props) {
  const isUser = role === 'user'

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content)
    } catch {
      // ignore
    }
  }

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} group`}>
      <div className={`flex items-start gap-4 max-w-2xl ${isUser ? 'ml-auto flex-row-reverse' : ''}`}>
        {!isUser && (
          <div className="flex-shrink-0 h-10 w-10 rounded-full bg-gradient-to-br from-blue-100 to-indigo-100 flex items-center justify-center font-semibold text-blue-700 border-2 border-white shadow-sm">
            <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
        )}
        <div className={`relative rounded-2xl p-5 shadow-lg backdrop-blur-sm border transition-all duration-200 group-hover:shadow-xl ${
          isUser 
            ? 'bg-gradient-to-br from-blue-600 via-blue-600 to-indigo-700 text-white border-blue-500/20 shadow-blue-500/20' 
            : 'bg-white/90 text-gray-800 border-gray-200/50 shadow-black/5'
        }`}>
          <div className="flex items-start gap-3">
            <div className="flex-1 prose prose-sm max-w-none break-words">
              <ReactMarkdown 
                components={{
                  code(props: any){
                    const { inline, children, ...rest } = props
                    return (
                      <code 
                        className={`rounded-md px-2 py-1 font-mono text-xs transition-colors ${
                          inline 
                            ? isUser 
                              ? 'bg-white/20 text-blue-100' 
                              : 'bg-gray-100 text-gray-800'
                            : isUser
                              ? 'block bg-white/20 text-blue-100 p-3'
                              : 'block bg-gray-100 text-gray-800 p-3'
                        }`} 
                        {...rest}
                      >
                        {children}
                      </code>
                    )
                  },
                  p: (props) => <p className="mb-2 last:mb-0 leading-relaxed" {...props} />,
                  ul: (props) => <ul className="list-disc list-inside space-y-1 mb-2" {...props} />,
                  ol: (props) => <ol className="list-decimal list-inside space-y-1 mb-2" {...props} />,
                  h1: (props) => <h1 className="text-lg font-bold mb-2" {...props} />,
                  h2: (props) => <h2 className="text-base font-bold mb-2" {...props} />,
                  h3: (props) => <h3 className="text-sm font-bold mb-1" {...props} />,
                }}
              >
                {content}
              </ReactMarkdown>
            </div>
            <button 
              onClick={handleCopy} 
              aria-label="Copy message" 
              className={`flex-shrink-0 p-2 rounded-lg transition-all duration-200 opacity-0 group-hover:opacity-100 ${
                isUser 
                  ? 'text-blue-100 hover:bg-white/20 hover:text-white' 
                  : 'text-gray-400 hover:bg-gray-100 hover:text-gray-600'
              }`}
            >
              <ClipboardCopy size={16} />
            </button>
          </div>
          
          {ui && (
            <div className="mt-4 pt-4 border-t border-gray-200/30">
              {ui}
            </div>
          )}
          
          {timestamp && (
            <div className={`mt-3 text-xs ${isUser ? 'text-blue-200' : 'text-gray-500'} flex items-center gap-2`}>
              <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
          )}
        </div>
        
        {isUser && (
          <div className="flex-shrink-0 h-10 w-10 rounded-full bg-gradient-to-br from-gray-700 to-gray-900 flex items-center justify-center font-semibold text-white border-2 border-white shadow-sm">
            <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
          </div>
        )}
      </div>
    </div>
  )
}
