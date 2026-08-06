import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["src/**/__tests__/**/*.test.ts"],
    // The GPU suite has its own config: it needs a browser and a real adapter,
    // and CI has neither. See vitest.gpu.config.ts.
    exclude: ["**/node_modules/**", "src/**/__tests__/gpu/**"],
  },
});
