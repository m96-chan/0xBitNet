/*
 * Minimal C example for oxbitnet FFI.
 *
 * Build:
 *   cargo build -p oxbitnet-ffi --release
 *   gcc examples/chat.c -I.. -L../../target/release -loxbitnet_ffi -o chat
 *
 * Run:
 *   LD_LIBRARY_PATH=../../target/release ./chat model.gguf "Hello, how are you?"
 */

#include <stdio.h>
#include "../oxbitnet.h"

/* Progress callback */
static void on_progress(const OxBitNetLoadProgress *p, void *userdata) {
    (void)userdata;
    const char *phases[] = {"Download", "Parse", "Upload"};
    const char *phase = (p->phase <= 2) ? phases[p->phase] : "?";
    fprintf(stderr, "\r[%s] %.1f%%", phase, p->fraction * 100.0);
    if (p->fraction >= 1.0)
        fprintf(stderr, "\n");
}

/* Token callback — print each token */
static int32_t on_token(const char *token, uintptr_t len, void *userdata) {
    (void)userdata;
    fwrite(token, 1, len, stdout);
    fflush(stdout);
    return 0; /* 0 = continue */
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model-path-or-url> <prompt>\n", argv[0]);
        return 1;
    }

    const char *source = argv[1];
    const char *user_prompt = argv[2];

    /* Load model */
    OxBitNetLoadOptions load_opts = oxbitnet_default_load_options();
    load_opts.on_progress = on_progress;

    fprintf(stderr, "Loading %s ...\n", source);
    OxBitNet *model = oxbitnet_load(source, &load_opts);
    if (!model) {
        const char *err = oxbitnet_error_message();
        fprintf(stderr, "Error: %s\n", err ? err : "unknown error");
        return 1;
    }
    fprintf(stderr, "Model loaded.\n");

    /* Build chat messages */
    OxBitNetChatMessage messages[] = {
        { .role = "user", .content = user_prompt },
    };

    OxBitNetGenerateOptions gen_opts = oxbitnet_default_generate_options();

    /* Generate */
    int32_t rc = oxbitnet_chat(
        model,
        messages,
        sizeof(messages) / sizeof(messages[0]),
        &gen_opts,
        on_token,
        NULL
    );

    printf("\n");

    if (rc != 0) {
        const char *err = oxbitnet_error_message();
        fprintf(stderr, "Generate error: %s\n", err ? err : "unknown error");
    }

    /* Cleanup */
    oxbitnet_free(model);
    return rc;
}
