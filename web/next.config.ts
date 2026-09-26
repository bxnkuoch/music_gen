import type { NextConfig } from "next";

// The Python API (FastAPI) runs separately; proxy /api/* to it so the browser only
// ever talks to this one origin (no CORS setup needed).
const apiUrl = process.env.API_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  rewrites() {
    return [{ source: "/api/:path*", destination: `${apiUrl}/api/:path*` }];
  },
};

export default nextConfig;
