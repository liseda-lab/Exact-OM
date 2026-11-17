import type { NextConfig } from "next";

// The production build is a static export served by exact-inspect, which decides per
// deployment profile which pages exist. `next dev` proxies /api to a running backend.
const backend = process.env.EXACT_DEV_BACKEND ?? "http://127.0.0.1:8000";
const isDev = process.env.NODE_ENV !== "production";

const nextConfig: NextConfig = {
  output: isDev ? undefined : "export",
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  ...(isDev
    ? {
        async rewrites() {
          return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
        },
      }
    : {}),
};

export default nextConfig;
