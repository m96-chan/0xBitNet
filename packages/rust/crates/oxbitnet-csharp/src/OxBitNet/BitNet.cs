using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OxBitNet
{
    /// <summary>
    /// A loaded BitNet model. Thread-safe and disposable.
    /// </summary>
    public sealed class BitNet : IDisposable
    {
        private IntPtr _handle;
        private readonly object _lock = new object();
        private bool _disposed;

        static BitNet()
        {
            NativeLoader.EnsureRegistered();
        }

        private BitNet(IntPtr handle)
        {
            _handle = handle;
        }

        /// <summary>
        /// Load a model synchronously (blocks the calling thread).
        /// </summary>
        public static BitNet LoadSync(string source, LoadOptions options = null)
        {
            byte[] sourceBytes = Encoding.UTF8.GetBytes(source + '\0');
            var sourcePin = GCHandle.Alloc(sourceBytes, GCHandleType.Pinned);

            try
            {
                var cOpts = NativeMethods.oxbitnet_default_load_options();
                GCHandle progressPin = default;
                GCHandle cacheDirPin = default;
                OxBitNetProgressFn progressDelegate = null;

                try
                {
                    if (options?.OnProgress != null)
                    {
                        progressDelegate = (ref OxBitNetLoadProgress progress, IntPtr userdata) =>
                        {
                            var p = new LoadProgress(
                                (LoadPhase)progress.phase,
                                progress.loaded,
                                progress.total,
                                progress.fraction);
                            options.OnProgress(p);
                        };
                        progressPin = GCHandle.Alloc(progressDelegate);
                        cOpts.on_progress = Marshal.GetFunctionPointerForDelegate(progressDelegate);
                    }

                    if (options?.CacheDir != null)
                    {
                        byte[] cacheDirBytes = Encoding.UTF8.GetBytes(options.CacheDir + '\0');
                        cacheDirPin = GCHandle.Alloc(cacheDirBytes, GCHandleType.Pinned);
                        cOpts.cache_dir = cacheDirPin.AddrOfPinnedObject();
                    }

                    IntPtr handle = NativeMethods.oxbitnet_load(sourcePin.AddrOfPinnedObject(), ref cOpts);
                    if (handle == IntPtr.Zero)
                    {
                        throw new OxBitNetException(GetErrorMessage() ?? "Failed to load model");
                    }

                    return new BitNet(handle);
                }
                finally
                {
                    if (progressPin.IsAllocated) progressPin.Free();
                    if (cacheDirPin.IsAllocated) cacheDirPin.Free();
                }
            }
            finally
            {
                sourcePin.Free();
            }
        }

        /// <summary>
        /// Load a model asynchronously.
        /// </summary>
        public static Task<BitNet> Load(string source, LoadOptions options = null)
        {
            return Task.Run(() => LoadSync(source, options));
        }

        /// <summary>
        /// Generate text from a raw prompt string. Tokens are delivered via callback.
        /// </summary>
        public void Generate(string prompt, Action<string> onToken, GenerateOptions options = null)
        {
            IntPtr handle = AcquireHandle();
            var cOpts = MakeGenerateOptions(options);

            byte[] promptBytes = Encoding.UTF8.GetBytes(prompt + '\0');
            var promptPin = GCHandle.Alloc(promptBytes, GCHandleType.Pinned);

            OxBitNetTokenFn tokenDelegate = (IntPtr token, nuint len, IntPtr userdata) =>
            {
                string str = Marshal.PtrToStringUTF8(token, (int)len);
                onToken(str);
                return 0;
            };
            var tokenPin = GCHandle.Alloc(tokenDelegate);

            try
            {
                int ret = NativeMethods.oxbitnet_generate(
                    handle,
                    promptPin.AddrOfPinnedObject(),
                    ref cOpts,
                    Marshal.GetFunctionPointerForDelegate(tokenDelegate),
                    IntPtr.Zero);

                if (ret != 0)
                {
                    throw new OxBitNetException(GetErrorMessage() ?? "Generate failed");
                }
            }
            finally
            {
                promptPin.Free();
                tokenPin.Free();
            }
        }

        /// <summary>
        /// Generate text from a raw prompt string asynchronously.
        /// </summary>
        public Task GenerateAsync(string prompt, Action<string> onToken, GenerateOptions options = null)
        {
            return Task.Run(() => Generate(prompt, onToken, options));
        }

        /// <summary>
        /// Generate text from chat messages. Tokens are delivered via callback.
        /// </summary>
        public void Chat(ChatMessage[] messages, Action<string> onToken, GenerateOptions options = null)
        {
            IntPtr handle = AcquireHandle();
            var cOpts = MakeGenerateOptions(options);

            // Marshal chat messages
            var cMessages = new OxBitNetChatMessage[messages.Length];
            var pins = new GCHandle[messages.Length * 2];
            int pinIdx = 0;

            try
            {
                for (int i = 0; i < messages.Length; i++)
                {
                    byte[] roleBytes = Encoding.UTF8.GetBytes(messages[i].Role + '\0');
                    byte[] contentBytes = Encoding.UTF8.GetBytes(messages[i].Content + '\0');

                    var rolePin = GCHandle.Alloc(roleBytes, GCHandleType.Pinned);
                    var contentPin = GCHandle.Alloc(contentBytes, GCHandleType.Pinned);
                    pins[pinIdx++] = rolePin;
                    pins[pinIdx++] = contentPin;

                    cMessages[i].role = rolePin.AddrOfPinnedObject();
                    cMessages[i].content = contentPin.AddrOfPinnedObject();
                }

                var messagesPin = GCHandle.Alloc(cMessages, GCHandleType.Pinned);

                OxBitNetTokenFn tokenDelegate = (IntPtr token, nuint len, IntPtr userdata) =>
                {
                    string str = Marshal.PtrToStringUTF8(token, (int)len);
                    onToken(str);
                    return 0;
                };
                var tokenPin = GCHandle.Alloc(tokenDelegate);

                try
                {
                    int ret = NativeMethods.oxbitnet_chat(
                        handle,
                        messagesPin.AddrOfPinnedObject(),
                        (nuint)messages.Length,
                        ref cOpts,
                        Marshal.GetFunctionPointerForDelegate(tokenDelegate),
                        IntPtr.Zero);

                    if (ret != 0)
                    {
                        throw new OxBitNetException(GetErrorMessage() ?? "Chat failed");
                    }
                }
                finally
                {
                    messagesPin.Free();
                    tokenPin.Free();
                }
            }
            finally
            {
                for (int i = 0; i < pinIdx; i++)
                {
                    if (pins[i].IsAllocated) pins[i].Free();
                }
            }
        }

        /// <summary>
        /// Generate text from chat messages asynchronously.
        /// </summary>
        public Task ChatAsync(ChatMessage[] messages, Action<string> onToken, GenerateOptions options = null)
        {
            return Task.Run(() => Chat(messages, onToken, options));
        }

        /// <summary>
        /// Release all GPU resources. Safe to call multiple times.
        /// Also called automatically by Dispose().
        /// </summary>
        public void Dispose()
        {
            lock (_lock)
            {
                if (!_disposed && _handle != IntPtr.Zero)
                {
                    NativeMethods.oxbitnet_free(_handle);
                    _handle = IntPtr.Zero;
                    _disposed = true;
                }
            }
        }

        private IntPtr AcquireHandle()
        {
            lock (_lock)
            {
                if (_disposed || _handle == IntPtr.Zero)
                    throw new ObjectDisposedException(nameof(BitNet));
                return _handle;
            }
        }

        private static OxBitNetGenerateOptions MakeGenerateOptions(GenerateOptions options)
        {
            if (options == null)
                return NativeMethods.oxbitnet_default_generate_options();

            return new OxBitNetGenerateOptions
            {
                max_tokens = (nuint)options.MaxTokens,
                temperature = options.Temperature,
                top_k = (nuint)options.TopK,
                repeat_penalty = options.RepeatPenalty,
                repeat_last_n = (nuint)options.RepeatLastN,
            };
        }

        private static string GetErrorMessage()
        {
            IntPtr ptr = NativeMethods.oxbitnet_error_message();
            if (ptr == IntPtr.Zero) return null;
            return Marshal.PtrToStringUTF8(ptr);
        }
    }
}
