using System;
using System.Runtime.InteropServices;

namespace OxBitNet
{
    internal enum OxBitNetLoadPhase : int
    {
        Download = 0,
        Parse = 1,
        Upload = 2,
    }

    internal enum OxBitNetLogLevel : byte
    {
        Trace = 0,
        Debug = 1,
        Info = 2,
        Warn = 3,
        Error = 4,
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct OxBitNetGenerateOptions
    {
        public nuint max_tokens;
        public float temperature;
        public nuint top_k;
        public float repeat_penalty;
        public nuint repeat_last_n;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct OxBitNetLoadProgress
    {
        public OxBitNetLoadPhase phase;
        public ulong loaded;
        public ulong total;
        public double fraction;
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void OxBitNetProgressFn(ref OxBitNetLoadProgress progress, IntPtr userdata);

    [StructLayout(LayoutKind.Sequential)]
    internal struct OxBitNetLoadOptions
    {
        public IntPtr on_progress; // OxBitNetProgressFn
        public IntPtr progress_userdata;
        public IntPtr cache_dir;   // const char*
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate int OxBitNetTokenFn(IntPtr token, nuint len, IntPtr userdata);

    [StructLayout(LayoutKind.Sequential)]
    internal struct OxBitNetChatMessage
    {
        public IntPtr role;    // const char*
        public IntPtr content; // const char*
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal delegate void OxBitNetLogFn(OxBitNetLogLevel level, IntPtr message, nuint len, IntPtr userdata);

    internal static class NativeMethods
    {
        private const string LibName = "oxbitnet_ffi";

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void oxbitnet_set_logger(IntPtr callback, IntPtr userdata, byte min_level);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern OxBitNetGenerateOptions oxbitnet_default_generate_options();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern OxBitNetLoadOptions oxbitnet_default_load_options();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr oxbitnet_load(IntPtr source, ref OxBitNetLoadOptions options);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr oxbitnet_load(IntPtr source, IntPtr options);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void oxbitnet_free(IntPtr model);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int oxbitnet_generate(
            IntPtr model,
            IntPtr prompt,
            ref OxBitNetGenerateOptions options,
            IntPtr callback,
            IntPtr userdata);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int oxbitnet_chat(
            IntPtr model,
            IntPtr messages,
            nuint num_messages,
            ref OxBitNetGenerateOptions options,
            IntPtr callback,
            IntPtr userdata);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr oxbitnet_error_message();
    }
}
