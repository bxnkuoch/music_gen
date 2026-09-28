import type { NextConfig } from "next";

// The Python API (FastAPI) runs separately; proxy /api/* to it so the browser only
// ever talks to this one origin (no CORS setup needed).
const apiUrl = process.env.API_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  // The proxy buffers request bodies and cuts them at 10 MB by default, which breaks
  // song uploads ("More like this"). Match the API's limit (references.MAX_BYTES).
  experimental: { proxyClientMaxBodySize: "50mb" },
  rewrites() {
    return [{ source: "/api/:path*", destination: `${apiUrl}/api/:path*` }];
  },
};

export default nextConfig;
