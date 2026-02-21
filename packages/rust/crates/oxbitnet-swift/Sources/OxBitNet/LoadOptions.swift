import Foundation

/// Phase of the model loading process.
public enum LoadPhase: Int, Sendable {
    case download = 0
    case parse = 1
    case upload = 2
}

/// Progress information during model loading.
public struct LoadProgress: Sendable {
    public let phase: LoadPhase
    public let loaded: UInt64
    public let total: UInt64
    public let fraction: Double
}

/// Options for loading a model.
public struct LoadOptions: Sendable {
    public var onProgress: (@Sendable (LoadProgress) -> Void)?
    public var cacheDir: String?

    public init(
        onProgress: (@Sendable (LoadProgress) -> Void)? = nil,
        cacheDir: String? = nil
    ) {
        self.onProgress = onProgress
        self.cacheDir = cacheDir
    }
}
