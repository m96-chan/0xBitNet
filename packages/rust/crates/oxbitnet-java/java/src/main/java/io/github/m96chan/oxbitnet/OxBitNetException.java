package io.github.m96chan.oxbitnet;

/**
 * Runtime exception thrown by oxbitnet operations (model loading, generation).
 */
public class OxBitNetException extends RuntimeException {

    public OxBitNetException(String message) {
        super(message);
    }

    public OxBitNetException(String message, Throwable cause) {
        super(message, cause);
    }
}
