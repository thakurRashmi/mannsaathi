import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Calming palette — muted, not clinical.
        sage: {
          50: "#f4f7f4",
          100: "#e6ede6",
          500: "#7a9b7e",
          700: "#4a6b4f",
        },
        warm: {
          50: "#faf7f2",
          100: "#f2ebdd",
        },
      },
    },
  },
  plugins: [],
};

export default config;
