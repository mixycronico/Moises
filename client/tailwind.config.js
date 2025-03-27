/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#6A11CB',
          light: '#8A64FF',
          dark: '#4B0F91'
        },
        secondary: {
          DEFAULT: '#4FFBDF',
          light: '#7DFFED',
          dark: '#30C5B1'
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      boxShadow: {
        'cosmic': '0 0 20px rgba(138, 100, 255, 0.3)',
        'neon': '0 0 10px rgba(79, 251, 223, 0.5)',
      },
      animation: {
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        }
      },
      backgroundImage: {
        'cosmic-gradient': 'linear-gradient(to right bottom, #6A11CB, #2575FC)',
        'secondary-gradient': 'linear-gradient(to right, #4FFBDF, #00CCCB)',
      }
    },
  },
  plugins: [],
}