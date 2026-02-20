import io.github.m96chan.oxbitnet.BitNet;
import io.github.m96chan.oxbitnet.ChatMessage;
import io.github.m96chan.oxbitnet.GenerateOptions;
import io.github.m96chan.oxbitnet.LoadOptions;

import java.util.List;

/**
 * Minimal chat example using oxbitnet Java bindings.
 *
 * <p>Usage:
 * <pre>
 *   cargo build -p oxbitnet-java --release
 *   javac -cp ../java/src/main/java:. Chat.java
 *   java -Djava.library.path=../../../../target/release -cp ../java/src/main/java:. Chat model.gguf "Hello!"
 * </pre>
 */
public class Chat {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: java Chat <model.gguf> <prompt>");
            System.exit(1);
        }

        String modelPath = args[0];
        String userMessage = args[1];

        System.err.println("Loading model: " + modelPath);

        try (BitNet model = BitNet.loadSync(modelPath, new LoadOptions()
                .onProgress((phase, loaded, total, fraction) ->
                    System.err.printf("\r[%s] %.1f%%", phase, fraction * 100)))) {

            System.err.println("\nModel loaded. Generating...\n");

            model.chat(
                List.of(new ChatMessage("user", userMessage)),
                token -> {
                    System.out.print(token);
                    return true;
                },
                new GenerateOptions().temperature(0.7f).maxTokens(512)
            );

            System.out.println();
        }
    }
}
