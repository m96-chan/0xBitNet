package io.github.m96chan.oxbitnet;

/**
 * Callback for streaming token output during text generation.
 */
@FunctionalInterface
public interface TokenCallback {

    /**
     * Called for each generated token.
     *
     * @param token the generated token string
     * @return {@code true} to continue generation, {@code false} to stop early
     */
    boolean onToken(String token);
}
