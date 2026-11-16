import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// The browser only ever talks to this server. Requests to the API paths are
// forwarded to zypherLL, and the API key is added here, on the way through,
// so it never reaches the page.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "ZYPHER_");
  const target = env.ZYPHER_API_URL || "http://127.0.0.1:8000";
  const key = env.ZYPHER_API_KEY || "";

  const forward = {
    target,
    changeOrigin: true,
    headers: key ? { Authorization: `Bearer ${key}` } : {},
  };

  const proxy = { "/v1": forward, "/health": forward };

  return {
    plugins: [react()],
    server: { host: "127.0.0.1", port: 5173, proxy },
    preview: { host: "127.0.0.1", port: 4173, proxy },
  };
});
