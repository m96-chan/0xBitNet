import Foundation

/// Errors thrown by OxBitNet operations.
public enum OxBitNetError: Error, LocalizedError, Sendable {
    case loadFailed(String)
    case generateFailed(String)
    case disposed

    public var errorDescription: String? {
        switch self {
        case .loadFailed(let msg): return "Load failed: \(msg)"
        case .generateFailed(let msg): return "Generate failed: \(msg)"
        case .disposed: return "Model has been disposed"
        }
    }
}
