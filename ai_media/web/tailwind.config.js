/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      backgroundColor: {
        primary: 'var(--bg-primary)',
        secondary: 'var(--bg-secondary)',
        tertiary: 'var(--bg-tertiary)',
        overlay: 'var(--bg-overlay)',
      },
      textColor: {
        primary: 'var(--text-primary)',
        secondary: 'var(--text-secondary)',
        tertiary: 'var(--text-tertiary)',
        // Map brand text to the brand palette
        brand: {
           400: '#38bdf8', // for nav-item.active
        }
      },
      borderColor: {
        DEFAULT: 'var(--border-color)',
        border: 'var(--border-color)',
      },
      colors: {
        // Keep primary as a general color if needed, but mapped to background usually
        primary: {
           DEFAULT: 'var(--bg-primary)',
           ...require('tailwindcss/colors').slate // fallback or just remove if not needed? 
           // actually, let's just leave the brand/slate here
        },
        // Preserve existing palette
        slate: {
           50: '#f8fafc',
           100: '#f1f5f9',
           200: '#e2e8f0',
           300: '#cbd5e1',
           400: '#94a3b8',
           500: '#64748b',
           600: '#475569',
           700: '#334155',
           800: '#1e293b',
           900: '#0f172a',
           950: '#020617',
        },
        brand: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
          950: '#082f49',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      keyframes: {
        'progress-bounce': {
          '0%': { left: '-35%', right: '100%' },
          '100%': { left: '100%', right: '-35%' },
        },
      },
      animation: {
        'progress-bounce': 'progress-bounce 1.5s linear infinite',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
