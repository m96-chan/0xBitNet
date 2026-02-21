package io.github.m96chan.oxbitnet;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

/**
 * Extracts and loads native libraries bundled inside the JAR.
 *
 * <p>Looks for the native library at
 * {@code META-INF/native/{classifier}/{libfile}} on the classpath, extracts it
 * to a temporary directory, and loads it via {@link System#load(String)}.
 * Falls back to {@link System#loadLibrary(String)} if the bundled library is
 * not found (e.g. when running from a development build).
 */
final class NativeLoader {

    private NativeLoader() {}

    /**
     * Load a native library by base name (e.g. "oxbitnet_java").
     *
     * @param baseName library base name without prefix/suffix
     */
    static void load(String baseName) {
        String classifier = detectClassifier();
        String libFile = mapLibraryName(baseName);
        String resourcePath = "META-INF/native/" + classifier + "/" + libFile;

        InputStream in = NativeLoader.class.getClassLoader().getResourceAsStream(resourcePath);
        if (in == null) {
            // Not bundled — fall back to java.library.path
            System.loadLibrary(baseName);
            return;
        }

        try {
            Path tempDir = Files.createTempDirectory("oxbitnet-native");
            Path tempLib = tempDir.resolve(libFile);
            Files.copy(in, tempLib, StandardCopyOption.REPLACE_EXISTING);
            in.close();
            tempLib.toFile().deleteOnExit();
            tempDir.toFile().deleteOnExit();
            System.load(tempLib.toAbsolutePath().toString());
        } catch (IOException e) {
            throw new RuntimeException("Failed to extract native library: " + resourcePath, e);
        }
    }

    private static String detectClassifier() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        String arch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);

        String osName;
        if (os.contains("linux")) {
            osName = "linux";
        } else if (os.contains("mac") || os.contains("darwin")) {
            osName = "macos";
        } else if (os.contains("win")) {
            osName = "windows";
        } else {
            throw new UnsupportedOperationException("Unsupported OS: " + os);
        }

        String archName;
        if (arch.equals("amd64") || arch.equals("x86_64")) {
            archName = "x86_64";
        } else if (arch.equals("aarch64") || arch.equals("arm64")) {
            archName = "aarch64";
        } else {
            throw new UnsupportedOperationException("Unsupported architecture: " + arch);
        }

        return osName + "-" + archName;
    }

    private static String mapLibraryName(String baseName) {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (os.contains("win")) {
            return baseName + ".dll";
        } else if (os.contains("mac") || os.contains("darwin")) {
            return "lib" + baseName + ".dylib";
        } else {
            return "lib" + baseName + ".so";
        }
    }
}
