import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Ericsson Hilda", system-ui, -apple-system, sans-serif'],
        mono: ['"Ericsson Hilda Mono", ui-monospace, monospace'],
      },
      colors: {
        eds: {
          blue: "#1174e6",
          "blue-dark": "#0d5bbd",
          "blue-light": "#e8f0fe",
          "blue-bg": "#f0f7ff",
          green: "#288964",
          "green-light": "#e8f5e9",
          "green-bg": "#f0faf4",
          orange: "#e66e19",
          "orange-light": "#fff3e0",
          red: "#dc2d37",
          "red-light": "#fce4ec",
          purple: "#8e45b0",
          "purple-light": "#f3e5f5",
          gray: {
            50: "#fafafa",
            100: "#f5f5f5",
            200: "#ebebeb",
            300: "#dcdcdc",
            400: "#b0b0b0",
            500: "#767676",
            600: "#4e4e4e",
            700: "#333333",
            800: "#242424",
            900: "#1a1a1a",
            950: "#181818",
          },
        },
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        success: {
          DEFAULT: "#288964",
          foreground: "#ffffff",
        },
        warning: {
          DEFAULT: "#e66e19",
          foreground: "#ffffff",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [],
} satisfies Config;
