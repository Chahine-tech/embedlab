import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // The engine writes artifacts to ../runs. Serving that directory as-is means
  // the UI reads exactly the files a user would hand to anything else, rather
  // than a shape prepared for it.
  publicDir: "public",
  server: { fs: { allow: [".."] } },
  optimizeDeps: {
    // duckdb-wasm ships its workers as separate entry points; excluding it
    // keeps Vite from trying to pre-bundle them into one chunk.
    exclude: ["@duckdb/duckdb-wasm"],
  },
});
