package io.github.m96chan.oxbitnet;

/**
 * A chat message with a role and content.
 *
 * @param role    the message role ("system", "user", or "assistant")
 * @param content the message text
 */
public record ChatMessage(String role, String content) {
}
