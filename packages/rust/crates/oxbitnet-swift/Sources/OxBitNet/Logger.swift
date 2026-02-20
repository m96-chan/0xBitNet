import Foundation
import COxBitNet

/// Log level for internal library messages.
public enum LogLevel: UInt8, Sendable {
    case trace = 0
    case debug = 1
    case info = 2
    case warn = 3
    case error = 4
}

/// Holds the logger callback, kept alive for the process lifetime.
private final class LoggerBox {
    let handler: @Sendable (LogLevel, String) -> Void

    init(_ handler: @escaping @Sendable (LogLevel, String) -> Void) {
        self.handler = handler
    }
}

/// Leaked reference to keep the logger box alive.
private var _loggerBox: Unmanaged<LoggerBox>?

/// Install a global logger that receives all internal log messages.
///
/// Must be called before any `BitNet.load` call. Can only be called once;
/// subsequent calls are no-ops.
///
/// - Parameters:
///   - minLevel: Minimum log level to receive (default: `.info`).
///   - handler: Callback receiving log level and message string.
public func setLogger(
    minLevel: LogLevel = .info,
    handler: @escaping @Sendable (LogLevel, String) -> Void
) {
    let box = LoggerBox(handler)
    let unmanaged = Unmanaged.passRetained(box)
    _loggerBox = unmanaged

    oxbitnet_set_logger(
        { (level, message, len, userdata) in
            guard let userdata = userdata, let message = message else { return }
            let box = Unmanaged<LoggerBox>.fromOpaque(userdata)
                .takeUnretainedValue()
            let str = String(
                bytesNoCopy: UnsafeMutableRawPointer(mutating: message),
                length: Int(len),
                encoding: .utf8,
                freeWhenDone: false
            ) ?? ""
            let swiftLevel = LogLevel(rawValue: UInt8(level.rawValue)) ?? .info
            box.handler(swiftLevel, str)
        },
        unmanaged.toOpaque(),
        minLevel.rawValue
    )
}
