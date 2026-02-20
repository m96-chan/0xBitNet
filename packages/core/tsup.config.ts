import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts", "src/internals.ts"],
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  treeshake: true,
  loader: {
    ".wgsl": "text",
  },
});
