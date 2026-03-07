using System;

namespace OxBitNet
{
    /// <summary>
    /// Exception thrown by OxBitNet operations.
    /// </summary>
    public class OxBitNetException : Exception
    {
        public OxBitNetException(string message) : base(message) { }
    }
}
