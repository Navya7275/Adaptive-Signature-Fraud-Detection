/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Brand accent
        accent: {
          DEFAULT: "#6366f1",  // indigo-500
          hover: "#4f46e5",
          light: "#818cf8",
        },
        // Dark theme surface
        surface: {
          DEFAULT: "#0b0b0f",
          card: "#141419",
          elevated: "#1c1c24",
          border: "#26262e",
        },
        // Status colors (used in both themes)
        success: "#10b981",
        warning: "#f59e0b",
        danger: "#ef4444",
        info: "#06b6d4",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "Menlo", "Monaco", "monospace"],
      },
      boxShadow: {
        "glow": "0 0 40px -10px rgba(99, 102, 241, 0.35)",
        "glow-lg": "0 0 60px -10px rgba(99, 102, 241, 0.4)",
        "float": "0 10px 40px -10px rgba(0, 0, 0, 0.2)",
        "float-lg": "0 20px 60px -12px rgba(0, 0, 0, 0.3)",
      },
      animation: {
        "fade-in": "fadeIn 0.3s ease-out",
        "slide-up": "slideUp 0.4s ease-out",
        "pulse-soft": "pulseSoft 2s ease-in-out infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: 0 },
          "100%": { opacity: 1 },
        },
        slideUp: {
          "0%": { opacity: 0, transform: "translateY(10px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
        pulseSoft: {
          "0%, 100%": { opacity: 1 },
          "50%": { opacity: 0.6 },
        },
      },
      backdropBlur: {
        xs: "2px",
      },
    },
  },
  plugins: [],
};