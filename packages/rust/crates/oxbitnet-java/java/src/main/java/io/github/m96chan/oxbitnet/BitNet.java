package io.github.m96chan.oxbitnet;

import java.util.List;

/**
 * Main entry point for BitNet inference.
 *
 * <p>Loads a BitNet b1.58 ternary LLM from a GGUF file and provides streaming
 * text generation via GPU (wgpu). Implements {@link AutoCloseable} for use with
 * try-with-resources.
 *
 * <pre>{@code
 * try (BitNet model = BitNet.loadSync("model.gguf")) {
 *     model.generate("Hello!", token -> {
 *         System.out.print(token);
 *         return true;
 *     });
 * }
 * }</pre>
 */
public class BitNet implements AutoCloseable {

    static {
        System.loadLibrary("oxbitnet_java");
    }

    private long nativeHandle;

    private BitNet(long handle) {
        this.nativeHandle = handle;
    }

    /**
     * Load a BitNet model from a GGUF file path or URL.
     *
     * @param source path or URL to a GGUF model file
     * @return a loaded BitNet model ready for inference
     * @throws OxBitNetException if loading fails
     */
    public static BitNet loadSync(String source) {
        long handle = nativeLoad(source, null);
        if (handle == 0) {
            throw new OxBitNetException("Failed to load model from: " + source);
        }
        return new BitNet(handle);
    }

    /**
     * Load a BitNet model with options (progress callback, cache dir).
     *
     * @param source  path or URL to a GGUF model file
     * @param options load options
     * @return a loaded BitNet model ready for inference
     * @throws OxBitNetException if loading fails
     */
    public static BitNet loadSync(String source, LoadOptions options) {
        long handle = nativeLoad(source, options);
        if (handle == 0) {
            throw new OxBitNetException("Failed to load model from: " + source);
        }
        return new BitNet(handle);
    }

    /**
     * Generate text from a raw prompt with streaming callback.
     *
     * @param prompt   the prompt string
     * @param callback receives each token; return true to continue, false to stop
     * @return number of tokens generated
     * @throws OxBitNetException if generation fails or model is disposed
     */
    public int generate(String prompt, TokenCallback callback) {
        return generate(prompt, callback, new GenerateOptions());
    }

    /**
     * Generate text from a raw prompt with streaming callback and options.
     *
     * @param prompt   the prompt string
     * @param callback receives each token; return true to continue, false to stop
     * @param options  generation options (temperature, maxTokens, etc.)
     * @return number of tokens generated
     * @throws OxBitNetException if generation fails or model is disposed
     */
    public int generate(String prompt, TokenCallback callback, GenerateOptions options) {
        checkNotDisposed();
        return nativeGenerate(
            nativeHandle, prompt,
            options.maxTokens, options.temperature,
            options.topK, options.repeatPenalty, options.repeatLastN,
            callback
        );
    }

    /**
     * Generate text from chat messages with streaming callback.
     *
     * @param messages list of chat messages
     * @param callback receives each token; return true to continue, false to stop
     * @return number of tokens generated
     * @throws OxBitNetException if generation fails or model is disposed
     */
    public int chat(List<ChatMessage> messages, TokenCallback callback) {
        return chat(messages, callback, new GenerateOptions());
    }

    /**
     * Generate text from chat messages with streaming callback and options.
     *
     * @param messages list of chat messages
     * @param callback receives each token; return true to continue, false to stop
     * @param options  generation options
     * @return number of tokens generated
     * @throws OxBitNetException if generation fails or model is disposed
     */
    public int chat(List<ChatMessage> messages, TokenCallback callback, GenerateOptions options) {
        checkNotDisposed();
        String[] roles = new String[messages.size()];
        String[] contents = new String[messages.size()];
        for (int i = 0; i < messages.size(); i++) {
            roles[i] = messages.get(i).role();
            contents[i] = messages.get(i).content();
        }
        return nativeChat(
            nativeHandle, roles, contents,
            options.maxTokens, options.temperature,
            options.topK, options.repeatPenalty, options.repeatLastN,
            callback
        );
    }

    /**
     * Release all GPU resources. Safe to call multiple times.
     */
    public void dispose() {
        if (nativeHandle != 0) {
            nativeFree(nativeHandle);
            nativeHandle = 0;
        }
    }

    /** Alias for {@link #dispose()}, enables try-with-resources. */
    @Override
    public void close() {
        dispose();
    }

    private void checkNotDisposed() {
        if (nativeHandle == 0) {
            throw new OxBitNetException("Model has been disposed");
        }
    }

    // -- Native methods --

    private static native long nativeLoad(String source, LoadOptions options);

    private static native int nativeGenerate(
        long handle, String prompt,
        long maxTokens, float temperature, long topK,
        float repeatPenalty, long repeatLastN,
        TokenCallback callback
    );

    private static native int nativeChat(
        long handle, String[] roles, String[] contents,
        long maxTokens, float temperature, long topK,
        float repeatPenalty, long repeatLastN,
        TokenCallback callback
    );

    private static native void nativeFree(long handle);
}
