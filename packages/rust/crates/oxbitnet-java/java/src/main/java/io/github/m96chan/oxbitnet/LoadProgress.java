package io.github.m96chan.oxbitnet;

/**
 * Immutable snapshot of model loading progress.
 *
 * @param phase    current phase ("download", "parse", or "upload")
 * @param loaded   bytes/items loaded so far
 * @param total    total bytes/items expected
 * @param fraction progress as a fraction (0.0 to 1.0)
 */
public record LoadProgress(String phase, long loaded, long total, double fraction) {
}
