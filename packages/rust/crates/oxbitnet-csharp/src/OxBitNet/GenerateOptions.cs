namespace OxBitNet
{
    /// <summary>
    /// Options for text generation.
    /// </summary>
    public sealed class GenerateOptions
    {
        /// <summary>Maximum number of tokens to generate.</summary>
        public int MaxTokens { get; set; } = 256;

        /// <summary>Sampling temperature.</summary>
        public float Temperature { get; set; } = 1.0f;

        /// <summary>Top-k sampling parameter.</summary>
        public int TopK { get; set; } = 50;

        /// <summary>Repetition penalty.</summary>
        public float RepeatPenalty { get; set; } = 1.1f;

        /// <summary>Window size for repetition penalty.</summary>
        public int RepeatLastN { get; set; } = 64;
    }
}
