using System;

namespace OxBitNet
{
    /// <summary>
    /// Options for loading a model.
    /// </summary>
    public sealed class LoadOptions
    {
        /// <summary>Progress callback invoked during model loading.</summary>
        public Action<LoadProgress> OnProgress { get; set; }

        /// <summary>Cache directory path (optional).</summary>
        public string CacheDir { get; set; }
    }
}
