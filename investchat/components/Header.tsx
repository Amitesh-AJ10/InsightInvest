"use client"

import Link from 'next/link'
import { ThemeToggle } from './theme-toggle'

export default function Header() {

  return (
    <header className="w-full border-b border-border bg-background/80 backdrop-blur-md shadow-sm">
      <div className="mx-auto max-w-7xl px-6 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-3 group">
          <div className="relative h-12 w-12 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-white font-bold shadow-lg group-hover:shadow-xl transition-shadow duration-200">
            <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
            <div className="absolute -inset-0.5 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl opacity-0 group-hover:opacity-20 transition-opacity duration-200"></div>
          </div>
          <div className="text-xl font-bold bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text text-transparent group-hover:from-blue-600 group-hover:to-indigo-600 transition-all duration-200">
            InsightInvest
          </div>
        </Link>
        
        <nav className="hidden md:flex items-center gap-8 text-sm font-medium">
          <a className="text-gray-600 hover:text-blue-600 transition-colors duration-200 relative group" href="#">
            Products
            <span className="absolute inset-x-0 -bottom-1 h-0.5 bg-blue-600 scale-x-0 group-hover:scale-x-100 transition-transform duration-200"></span>
          </a>
          <a className="text-gray-600 hover:text-blue-600 transition-colors duration-200 relative group" href="#">
            Documentation
            <span className="absolute inset-x-0 -bottom-1 h-0.5 bg-blue-600 scale-x-0 group-hover:scale-x-100 transition-transform duration-200"></span>
          </a>
          <a className="text-gray-600 hover:text-blue-600 transition-colors duration-200 relative group" href="#">
            About
            <span className="absolute inset-x-0 -bottom-1 h-0.5 bg-blue-600 scale-x-0 group-hover:scale-x-100 transition-transform duration-200"></span>
          </a>
        </nav>
        
        <div className="flex items-center gap-4">
          <ThemeToggle />
          
          <a 
            href="#" 
            className="hidden sm:inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold rounded-lg shadow-lg hover:shadow-xl transition-all duration-200 hover:-translate-y-0.5"
          >
            <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Get Started
          </a>
          
          <div className="hidden md:flex items-center gap-3">
            <div className="h-10 w-10 rounded-full bg-gradient-to-br from-gray-700 to-gray-900 flex items-center justify-center text-white font-semibold shadow-lg border-2 border-white">
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
