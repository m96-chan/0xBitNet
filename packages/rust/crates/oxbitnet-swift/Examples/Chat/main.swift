import Foundation
import OxBitNet

@main
struct ChatExample {
    static func main() async throws {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print("Usage: Chat <model-path> [prompt]")
            Foundation.exit(1)
        }

        let modelPath = args[1]
        let prompt = args.count >= 3 ? args[2] : "Hello!"

        // Optional: enable logging
        setLogger(minLevel: .info) { level, message in
            print("[\(level)] \(message)", to: &standardError)
        }

        // Load model with progress
        print("Loading \(modelPath)...", to: &standardError)
        let model = try await BitNet.load(modelPath, options: LoadOptions(
            onProgress: { p in
                let pct = String(format: "%.1f", p.fraction * 100)
                print("  [\(p.phase)] \(pct)%", to: &standardError)
            }
        ))
        print("Model loaded.", to: &standardError)

        // Chat
        let messages: [ChatMessage] = [
            .user(prompt),
        ]

        for try await token in model.chat(messages) {
            print(token, terminator: "")
            fflush(stdout)
        }
        print()

        model.dispose()
    }
}

/// Helper to write to stderr.
private struct StderrOutputStream: TextOutputStream {
    mutating func write(_ string: String) {
        FileHandle.standardError.write(Data(string.utf8))
    }
}

private var standardError = StderrOutputStream()
