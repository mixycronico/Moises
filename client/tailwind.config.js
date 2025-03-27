/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
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