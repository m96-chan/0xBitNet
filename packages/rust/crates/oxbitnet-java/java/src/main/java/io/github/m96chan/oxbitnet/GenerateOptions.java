package io.github.m96chan.oxbitnet;

/**
 * Options for text generation. Uses a builder pattern — all setters return
 * {@code this} for chaining.
 *
 * <pre>{@code
 * new GenerateOptions()
 *     .temperature(0.7f)
 *     .maxTokens(512)
 *     .topK(40)
 * }</pre>
 */
public class GenerateOptions {

    long maxTokens = 256;
    float temperature = 1.0f;
    long topK = 50;
    float repeatPenalty = 1.1f;
    long repeatLastN = 64;

    public GenerateOptions maxTokens(long maxTokens) {
        this.maxTokens = maxTokens;
        return this;
    }

    public GenerateOptions temperature(float temperature) {
        this.temperature = temperature;
        return this;
    }

    public GenerateOptions topK(long topK) {
        this.topK = topK;
        return this;
    }

    public GenerateOptions repeatPenalty(float repeatPenalty) {
        this.repeatPenalty = repeatPenalty;
        return this;
    }

    public GenerateOptions repeatLastN(long repeatLastN) {
        this.repeatLastN = repeatLastN;
        return this;
    }
}
