namespace OxBitNet
{
    /// <summary>
    /// Phase of the model loading process.
    /// </summary>
    public enum LoadPhase
    {
        Download = 0,
        Parse = 1,
        Upload = 2,
    }

    /// <summary>
    /// Progress information during model loading.
    /// </summary>
    public sealed class LoadProgress
    {
        public LoadPhase Phase { get; }
        public ulong Loaded { get; }
        public ulong Total { get; }
        public double Fraction { get; }

        internal LoadProgress(LoadPhase phase, ulong loaded, ulong total, double fraction)
        {
            Phase = phase;
            Loaded = loaded;
            Total = total;
            Fraction = fraction;
        }
    }
}
