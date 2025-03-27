/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  safelist: [
    { pattern: /^bg-cosmic-/ },
    { pattern: /^text-cosmic-/ },
    { pattern: /^border-cosmic-/ },
    { pattern: /^from-cosmic-/ },
    { pattern: /^to-cosmic-/ },
    { pattern: /^via-cosmic-/ },
    'opacity-10', 'opacity-20', 'opacity-30', 'opacity-40', 'opacity-50', 'opacity-60', 'opacity-70', 'opacity-80', 'opacity-90'
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        display: ['Orbitron', 'sans-serif'],
      },
      colors: {
        cosmic: {
          dark: 'var(--cosmic-dark)',
          darkest: 'var(--cosmic-darkest)',
          primary: 'var(--cosmic-primary)',
          'primary-10': 'rgba(42, 42, 74, 0.1)',
          'primary-20': 'rgba(42, 42, 74, 0.2)',
          'primary-30': 'rgba(42, 42, 74, 0.3)',
          'primary-50': 'rgba(42, 42, 74, 0.5)',
          secondary: 'var(--cosmic-secondary)',
          accent: 'var(--cosmic-accent)',
          highlight: 'var(--cosmic-highlight)',
          glow: 'var(--cosmic-glow)',
          blue: 'var(--cosmic-blue)',
          green: 'var(--cosmic-green)',
          yellow: 'var(--cosmic-yellow)',
          red: 'var(--cosmic-red)',
        },
      },
      backgroundImage: {
        'cosmic-gradient': 'linear-gradient(135deg, var(--cosmic-darkest) 0%, var(--cosmic-dark) 100%)',
        'cosmic-accent-gradient': 'linear-gradient(90deg, var(--cosmic-accent) 0%, var(--cosmic-highlight) 100%)',
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'ping-slow': 'ping 3s cubic-bezier(0, 0, 0.2, 1) infinite',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [],
};