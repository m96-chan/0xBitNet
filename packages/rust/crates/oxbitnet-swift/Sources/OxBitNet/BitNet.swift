import Foundation
import COxBitNet

/// A loaded BitNet model. Thread-safe; blocking FFI calls are dispatched
/// to a background GCD queue to avoid starving the Swift cooperative pool.
public final class BitNet: @unchecked Sendable {
    private var handle: OpaquePointer?
    private let lock = NSLock()

    private init(handle: OpaquePointer) {
        self.handle = handle
    }

    deinit {
        dispose()
    }

    // MARK: - Loading

    /// Load a model asynchronously. The blocking C call runs on a background
    /// GCD queue so it won't block the Swift cooperative thread pool.
    public static func load(
        _ source: String,
        options: LoadOptions = LoadOptions()
    ) async throws -> BitNet {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let model = try loadSync(source, options: options)
                    continuation.resume(returning: model)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Load a model synchronously (blocks the calling thread).
    public static func loadSync(
        _ source: String,
        options: LoadOptions = LoadOptions()
    ) throws -> BitNet {
        var cOpts = oxbitnet_default_load_options()
        var progressBox: Unmanaged<ProgressBox>?
        var cacheDirCString: UnsafeMutablePointer<CChar>?

        // Set up progress callback
        if let onProgress = options.onProgress {
            let box = ProgressBox(onProgress)
            let unmanaged = Unmanaged.passRetained(box)
            progressBox = unmanaged
            cOpts.on_progress = progressCallback
            cOpts.progress_userdata = unmanaged.toOpaque()
        }

        // Set up cache dir
        if let cacheDir = options.cacheDir {
            cacheDirCString = strdup(cacheDir)
            cOpts.cache_dir = UnsafePointer(cacheDirCString)
        }

        defer {
            progressBox?.release()
            free(cacheDirCString)
        }

        guard let handle = oxbitnet_load(source, &cOpts) else {
            let msg = errorMessage() ?? "unknown error"
            throw OxBitNetError.loadFailed(msg)
        }

        return BitNet(handle: handle)
    }

    // MARK: - Generate

    /// Generate tokens from a raw prompt. Returns an `AsyncThrowingStream`
    /// that yields tokens as they are produced.
    public func generate(
        _ prompt: String,
        options: GenerateOptions = GenerateOptions()
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                do {
                    let handle = try acquireHandle()

                    var cOpts = OxBitNetGenerateOptions(
                        max_tokens: UInt(options.maxTokens),
                        temperature: options.temperature,
                        top_k: UInt(options.topK),
                        repeat_penalty: options.repeatPenalty,
                        repeat_last_n: UInt(options.repeatLastN)
                    )

                    let ctx = TokenStreamContext(continuation)
                    let unmanaged = Unmanaged.passRetained(ctx)
                    defer { unmanaged.release() }

                    let ret = oxbitnet_generate(
                        handle,
                        prompt,
                        &cOpts,
                        tokenCallback,
                        unmanaged.toOpaque()
                    )

                    if ret != 0 {
                        let msg = errorMessage() ?? "unknown error"
                        continuation.finish(throwing: OxBitNetError.generateFailed(msg))
                    } else {
                        continuation.finish()
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Chat

    /// Generate tokens from chat messages. Returns an `AsyncThrowingStream`
    /// that yields tokens as they are produced.
    public func chat(
        _ messages: [ChatMessage],
        options: GenerateOptions = GenerateOptions()
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                do {
                    let handle = try acquireHandle()

                    var cOpts = OxBitNetGenerateOptions(
                        max_tokens: UInt(options.maxTokens),
                        temperature: options.temperature,
                        top_k: UInt(options.topK),
                        repeat_penalty: options.repeatPenalty,
                        repeat_last_n: UInt(options.repeatLastN)
                    )

                    let ctx = TokenStreamContext(continuation)
                    let unmanaged = Unmanaged.passRetained(ctx)
                    defer { unmanaged.release() }

                    // Build C message array — all strings must stay alive
                    // through the FFI call.
                    let cMessages: [OxBitNetChatMessage] = messages.map { msg in
                        OxBitNetChatMessage(
                            role: (msg.role as NSString).utf8String,
                            content: (msg.content as NSString).utf8String
                        )
                    }

                    let ret = cMessages.withUnsafeBufferPointer { buf in
                        oxbitnet_chat(
                            handle,
                            buf.baseAddress,
                            UInt(buf.count),
                            &cOpts,
                            tokenCallback,
                            unmanaged.toOpaque()
                        )
                    }

                    if ret != 0 {
                        let msg = errorMessage() ?? "unknown error"
                        continuation.finish(throwing: OxBitNetError.generateFailed(msg))
                    } else {
                        continuation.finish()
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Dispose

    /// Release all GPU resources. Safe to call multiple times.
    /// Also called automatically by `deinit`.
    public func dispose() {
        lock.lock()
        defer { lock.unlock() }
        if let h = handle {
            oxbitnet_free(h)
            handle = nil
        }
    }

    // MARK: - Private

    private func acquireHandle() throws -> OpaquePointer {
        lock.lock()
        defer { lock.unlock() }
        guard let h = handle else {
            throw OxBitNetError.disposed
        }
        return h
    }

    private static func errorMessage() -> String? {
        guard let ptr = oxbitnet_error_message() else { return nil }
        return String(cString: ptr)
    }
}

// Non-static helper for error messages after FFI calls on the same thread.
private func errorMessage() -> String? {
    guard let ptr = oxbitnet_error_message() else { return nil }
    return String(cString: ptr)
}
