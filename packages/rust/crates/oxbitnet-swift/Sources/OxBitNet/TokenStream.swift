import Foundation
import COxBitNet

/// Holds the stream continuation, passed as C userdata via Unmanaged.
final class TokenStreamContext {
    let continuation: AsyncThrowingStream<String, Error>.Continuation

    init(_ continuation: AsyncThrowingStream<String, Error>.Continuation) {
        self.continuation = continuation
    }
}

/// C-compatible token callback that yields tokens into an AsyncThrowingStream.
///
/// Returns 0 to continue, non-zero to stop (when the stream is cancelled).
func tokenCallback(
    token: UnsafePointer<CChar>?,
    len: UInt,
    userdata: UnsafeMutableRawPointer?
) -> Int32 {
    guard let userdata = userdata, let token = token else { return 0 }

    let ctx = Unmanaged<TokenStreamContext>.fromOpaque(userdata)
        .takeUnretainedValue()

    let str = String(
        bytesNoCopy: UnsafeMutableRawPointer(mutating: token),
        length: Int(len),
        encoding: .utf8,
        freeWhenDone: false
    ) ?? String(cString: token)

    let result = ctx.continuation.yield(str)
    switch result {
    case .terminated:
        return 1  // signal cancellation to C
    default:
        return 0
    }
}

/// Holds the progress callback, passed as C userdata via Unmanaged.
final class ProgressBox {
    let handler: @Sendable (LoadProgress) -> Void

    init(_ handler: @escaping @Sendable (LoadProgress) -> Void) {
        self.handler = handler
    }
}

/// C-compatible progress callback that forwards to a Swift closure.
func progressCallback(
    progress: UnsafePointer<OxBitNetLoadProgress>?,
    userdata: UnsafeMutableRawPointer?
) {
    guard let progress = progress, let userdata = userdata else { return }

    let box = Unmanaged<ProgressBox>.fromOpaque(userdata)
        .takeUnretainedValue()

    let p = progress.pointee
    let phase = LoadPhase(rawValue: Int(p.phase.rawValue)) ?? .download

    box.handler(LoadProgress(
        phase: phase,
        loaded: p.loaded,
        total: p.total,
        fraction: p.fraction
    ))
}
