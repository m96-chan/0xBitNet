using System;
using System.Collections.Generic;
using OxBitNet;

class Program
{
    static void Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine("Usage: ChatConsole <model-path>");
            Environment.Exit(1);
        }

        string modelPath = args[0];

        Console.Error.WriteLine($"Loading {modelPath}...");
        using var model = BitNet.LoadSync(modelPath, new LoadOptions
        {
            OnProgress = p =>
            {
                Console.Error.WriteLine($"  [{p.Phase}] {p.Fraction * 100:F1}%");
            }
        });
        Console.Error.WriteLine("Model loaded.");

        var history = new List<ChatMessage>();
        Console.WriteLine("Type a message (or 'quit' to exit):");

        while (true)
        {
            Console.Write("\n> ");
            string input = Console.ReadLine();
            if (input == null || input.Trim().Equals("quit", StringComparison.OrdinalIgnoreCase))
                break;

            if (string.IsNullOrWhiteSpace(input))
                continue;

            history.Add(ChatMessage.User(input));

            Console.Write("\n");
            var response = new System.Text.StringBuilder();

            model.Chat(history.ToArray(), token =>
            {
                Console.Write(token);
                response.Append(token);
            }, new GenerateOptions { Temperature = 0.7f });

            Console.WriteLine();
            history.Add(ChatMessage.Assistant(response.ToString()));
        }
    }
}
