import Foundation

/// Options for text generation.
public struct GenerateOptions: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topK: Int
    public var repeatPenalty: Float
    public var repeatLastN: Int

    public init(
        maxTokens: Int = 256,
        temperature: Float = 1.0,
        topK: Int = 50,
        repeatPenalty: Float = 1.1,
        repeatLastN: Int = 64
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topK = topK
        self.repeatPenalty = repeatPenalty
        self.repeatLastN = repeatLastN
    }
}
