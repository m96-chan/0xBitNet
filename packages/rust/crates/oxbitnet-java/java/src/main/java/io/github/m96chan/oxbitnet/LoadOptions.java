package io.github.m96chan.oxbitnet;

/**
 * Options for loading a BitNet model.
 *
 * <pre>{@code
 * new LoadOptions()
 *     .onProgress((phase, loaded, total, fraction) ->
 *         System.err.printf("[%s] %.1f%%\n", phase, fraction * 100))
 *     .cacheDir("/tmp/oxbitnet-cache")
 * }</pre>
 */
public class LoadOptions {

    /** Progress callback (accessed from native code). */
    ProgressCallback onProgress;

    /** Cache directory path (accessed from native code). */
    String cacheDir;

    public LoadOptions onProgress(ProgressCallback callback) {
        this.onProgress = callback;
        return this;
    }

    public LoadOptions cacheDir(String path) {
        this.cacheDir = path;
        return this;
    }

    /**
     * Callback invoked during model loading to report progress.
     */
    @FunctionalInterface
    public interface ProgressCallback {

        /**
         * Called with progress information.
         *
         * @param phase    current phase ("download", "parse", or "upload")
         * @param loaded   bytes/items loaded so far
         * @param total    total bytes/items expected
         * @param fraction progress as a fraction (0.0 to 1.0)
         */
        void onProgress(String phase, long loaded, long total, double fraction);
    }
}
