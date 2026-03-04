using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace OxBitNet
{
    internal static class NativeLoader
    {
        private static bool _registered;

        internal static void EnsureRegistered()
        {
            if (_registered) return;
            _registered = true;

            NativeLibrary.SetDllImportResolver(typeof(NativeLoader).Assembly, Resolver);
        }

        private static IntPtr Resolver(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (libraryName != "oxbitnet_ffi")
                return IntPtr.Zero;

            string rid = GetRuntimeIdentifier();
            string fileName = GetNativeFileName();

            // Try: runtimes/{rid}/native/{lib}  (NuGet layout)
            string assemblyDir = Path.GetDirectoryName(assembly.Location) ?? ".";
            string nugetPath = Path.Combine(assemblyDir, "runtimes", rid, "native", fileName);
            if (NativeLibrary.TryLoad(nugetPath, out IntPtr handle))
                return handle;

            // Try: same directory as assembly
            string localPath = Path.Combine(assemblyDir, fileName);
            if (NativeLibrary.TryLoad(localPath, out handle))
                return handle;

            // Fallback: system search path
            if (NativeLibrary.TryLoad(libraryName, assembly, searchPath, out handle))
                return handle;

            return IntPtr.Zero;
        }

        private static string GetRuntimeIdentifier()
        {
            string os;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                os = "linux";
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                os = "osx";
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                os = "win";
            else
                os = "unknown";

            string arch = RuntimeInformation.OSArchitecture switch
            {
                Architecture.X64 => "x64",
                Architecture.Arm64 => "arm64",
                _ => "x64",
            };

            return $"{os}-{arch}";
        }

        private static string GetNativeFileName()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return "oxbitnet_ffi.dll";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                return "liboxbitnet_ffi.dylib";
            return "liboxbitnet_ffi.so";
        }
    }
}
