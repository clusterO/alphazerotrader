/** @type {import('next').NextConfig} */
const nextConfig = {
  swcMinify: false, // Disable SWC minifier to avoid SIGBUS/memory mapping issues
  eslint: {
    ignoreDuringBuilds: true, // Speed up build
  },
  typescript: {
    ignoreBuildErrors: true, // Speed up build
  }
};

export default nextConfig;
