import { defineConfig } from "vitest/config";

/**
 * Separate from vitest.config.ts on purpose. These tests need a GPU, CI runs on
 * ubuntu-latest without one, and the suite that gates pull requests should be
 * the one that can actually run there.
 */
export default defineConfig({
  test: {
    include: ["src/**/__tests__/gpu/**/*.gpu.test.ts"],
    testTimeout: 60_000,
    hookTimeout: 120_000,
    fileParallelism: false,
  },
});
