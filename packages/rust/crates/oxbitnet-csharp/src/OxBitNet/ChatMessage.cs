namespace OxBitNet
{
    /// <summary>
    /// A chat message with a role and content.
    /// </summary>
    public sealed class ChatMessage
    {
        public string Role { get; }
        public string Content { get; }

        public ChatMessage(string role, string content)
        {
            Role = role;
            Content = content;
        }

        public static ChatMessage System(string content) => new ChatMessage("system", content);
        public static ChatMessage User(string content) => new ChatMessage("user", content);
        public static ChatMessage Assistant(string content) => new ChatMessage("assistant", content);
    }
}
