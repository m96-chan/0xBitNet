//! C FFI bindings for oxbitnet.
//!
//! Provides a stable C ABI for loading BitNet models and generating text via
//! callback-based streaming. Intended as the shared foundation for Java/Android,
//! Swift/iOS, C#, and Haskell bindings.

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

use futures::StreamExt;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque handle to a loaded BitNet model.
pub struct OxBitNet {
    inner: Option<oxbitnet::BitNet>,
    runtime: tokio::runtime::Runtime,
}

// ---------------------------------------------------------------------------
// Thread-local error
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(e: impl std::fmt::Display) {
    let msg = CString::new(e.to_string()).unwrap_or_default();
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = Some(msg);
    });
}

fn clear_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

// ---------------------------------------------------------------------------
// C-visible types
// ---------------------------------------------------------------------------

/// Load progress phase.
#[repr(C)]
pub enum OxBitNetLoadPhase {
    Download = 0,
    Parse = 1,
    Upload = 2,
}

/// Progress information passed to the load progress callback.
#[repr(C)]
pub struct OxBitNetLoadProgress {
    pub phase: OxBitNetLoadPhase,
    pub loaded: u64,
    pub total: u64,
    pub fraction: f64,
}

/// Progress callback type. Called during model loading.
pub type OxBitNetProgressFn =
    Option<unsafe extern "C" fn(progress: *const OxBitNetLoadProgress, userdata: *mut std::ffi::c_void)>;

/// Options for loading a model.
#[repr(C)]
pub struct OxBitNetLoadOptions {
    /// Progress callback (nullable).
    pub on_progress: OxBitNetProgressFn,
    /// User data passed to the progress callback (nullable).
    pub progress_userdata: *mut std::ffi::c_void,
    /// Cache directory path, null-terminated UTF-8 (nullable).
    pub cache_dir: *const c_char,
}

/// Token callback type. Return 0 to continue, non-zero to stop.
pub type OxBitNetTokenFn =
    Option<unsafe extern "C" fn(token: *const c_char, len: usize, userdata: *mut std::ffi::c_void) -> i32>;

/// Options for text generation.
#[repr(C)]
pub struct OxBitNetGenerateOptions {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature.
    pub temperature: f32,
    /// Top-k sampling parameter.
    pub top_k: usize,
    /// Repetition penalty.
    pub repeat_penalty: f32,
    /// Window size for repetition penalty.
    pub repeat_last_n: usize,
}

/// A chat message with role and content.
#[repr(C)]
pub struct OxBitNetChatMessage {
    /// Role string, null-terminated UTF-8 (e.g. "user", "assistant", "system").
    pub role: *const c_char,
    /// Content string, null-terminated UTF-8.
    pub content: *const c_char,
}

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

/// Log level for the logger callback.
#[repr(C)]
pub enum OxBitNetLogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
}

/// Logger callback type.
pub type OxBitNetLogFn = unsafe extern "C" fn(
    level: OxBitNetLogLevel,
    message: *const c_char,
    len: usize,
    userdata: *mut std::ffi::c_void,
);

struct LoggerState {
    callback: OxBitNetLogFn,
    userdata: usize, // stored as usize for Send+Sync
}

// Safety: the caller guarantees the userdata pointer (and callback) are safe
// to call from any thread.
unsafe impl Send for LoggerState {}
unsafe impl Sync for LoggerState {}

static LOGGER: OnceLock<LoggerState> = OnceLock::new();
static MIN_LOG_LEVEL: AtomicU8 = AtomicU8::new(2); // default: Info

/// A tracing layer that forwards events to the C logger callback.
struct FfiLayer;

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for FfiLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let Some(state) = LOGGER.get() else {
            return;
        };

        let (level, c_level) = match *event.metadata().level() {
            tracing::Level::TRACE => (0u8, OxBitNetLogLevel::Trace),
            tracing::Level::DEBUG => (1, OxBitNetLogLevel::Debug),
            tracing::Level::INFO => (2, OxBitNetLogLevel::Info),
            tracing::Level::WARN => (3, OxBitNetLogLevel::Warn),
            tracing::Level::ERROR => (4, OxBitNetLogLevel::Error),
        };

        if level < MIN_LOG_LEVEL.load(Ordering::Relaxed) {
            return;
        }

        // Format the event message
        let mut buf = String::new();
        let mut visitor = MessageVisitor(&mut buf);
        event.record(&mut visitor);

        if let Ok(c_str) = CString::new(buf) {
            let len = c_str.as_bytes().len();
            unsafe {
                (state.callback)(
                    c_level,
                    c_str.as_ptr(),
                    len,
                    state.userdata as *mut std::ffi::c_void,
                );
            }
        }
    }
}

struct MessageVisitor<'a>(&'a mut String);

impl tracing::field::Visit for MessageVisitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write;
        if field.name() == "message" {
            let _ = write!(self.0, "{:?}", value);
        } else {
            if !self.0.is_empty() {
                self.0.push(' ');
            }
            let _ = write!(self.0, "{}={:?}", field.name(), value);
        }
    }
}

/// Install a logger callback that receives all internal log messages.
///
/// Must be called before `oxbitnet_load`. Can only be called once; subsequent
/// calls are no-ops. Pass `min_level` to filter: 0=Trace, 1=Debug, 2=Info,
/// 3=Warn, 4=Error.
///
/// # Safety
///
/// - `callback` must be a valid function pointer safe to call from any thread.
/// - `userdata` must remain valid for the lifetime of the process (or until no
///   more log messages will be emitted).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxbitnet_set_logger(
    callback: OxBitNetLogFn,
    userdata: *mut std::ffi::c_void,
    min_level: u8,
) {
    MIN_LOG_LEVEL.store(min_level.min(4), Ordering::Relaxed);

    if LOGGER
        .set(LoggerState {
            callback,
            userdata: userdata as usize,
        })
        .is_ok()
    {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;

        let _ = tracing_subscriber::registry().with(FfiLayer).try_init();
    }
}

// ---------------------------------------------------------------------------
// Default options
// ---------------------------------------------------------------------------

/// Return default generation options.
#[unsafe(no_mangle)]
pub extern "C" fn oxbitnet_default_generate_options() -> OxBitNetGenerateOptions {
    OxBitNetGenerateOptions {
        max_tokens: 256,
        temperature: 1.0,
        top_k: 50,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
    }
}

/// Return default load options (no progress callback, no cache dir).
#[unsafe(no_mangle)]
pub extern "C" fn oxbitnet_default_load_options() -> OxBitNetLoadOptions {
    OxBitNetLoadOptions {
        on_progress: None,
        progress_userdata: ptr::null_mut(),
        cache_dir: ptr::null(),
    }
}

// ---------------------------------------------------------------------------
// Load / Free
// ---------------------------------------------------------------------------

/// Load a BitNet model from a URL or local file path.
///
/// `source` must be a null-terminated UTF-8 string.
/// `options` may be NULL for defaults.
///
/// Returns an opaque handle on success, or NULL on error.
/// On error, call `oxbitnet_error_message()` for details.
///
/// # Safety
///
/// - `source` must be a valid null-terminated UTF-8 string.
/// - `options`, if non-null, must point to a valid `OxBitNetLoadOptions`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxbitnet_load(
    source: *const c_char,
    options: *const OxBitNetLoadOptions,
) -> *mut OxBitNet {
    clear_error();

    // Validate source
    if source.is_null() {
        set_error("source must not be NULL");
        return ptr::null_mut();
    }
    let source_str = match unsafe { CStr::from_ptr(source) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(format!("invalid UTF-8 in source: {e}"));
            return ptr::null_mut();
        }
    };

    // Build LoadOptions
    let mut load_opts = oxbitnet::LoadOptions::default();

    if !options.is_null() {
        let opts = unsafe { &*options };

        // Cache dir
        if !opts.cache_dir.is_null() {
            if let Ok(s) = unsafe { CStr::from_ptr(opts.cache_dir) }.to_str() {
                load_opts.cache_dir = Some(PathBuf::from(s));
            }
        }

        // Progress callback
        if let Some(cb) = opts.on_progress {
            // Cast to usize to satisfy Send bound on the closure.
            // Safety: the caller guarantees the userdata pointer remains valid.
            let userdata_addr = opts.progress_userdata as usize;
            load_opts.on_progress = Some(Box::new(move |p: oxbitnet::LoadProgress| {
                let phase = match p.phase {
                    oxbitnet::model::loader::LoadPhase::Download => OxBitNetLoadPhase::Download,
                    oxbitnet::model::loader::LoadPhase::Parse => OxBitNetLoadPhase::Parse,
                    oxbitnet::model::loader::LoadPhase::Upload => OxBitNetLoadPhase::Upload,
                };
                let c_progress = OxBitNetLoadProgress {
                    phase,
                    loaded: p.loaded,
                    total: p.total,
                    fraction: p.fraction,
                };
                unsafe { cb(&c_progress, userdata_addr as *mut std::ffi::c_void) };
            }));
        }
    }

    // Create runtime and load
    let runtime = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            set_error(format!("failed to create tokio runtime: {e}"));
            return ptr::null_mut();
        }
    };

    match runtime.block_on(oxbitnet::BitNet::load(source_str, load_opts)) {
        Ok(bitnet) => Box::into_raw(Box::new(OxBitNet {
            inner: Some(bitnet),
            runtime,
        })),
        Err(e) => {
            set_error(e);
            ptr::null_mut()
        }
    }
}

/// Free a model handle and release all GPU resources.
///
/// Passing NULL is a no-op.
///
/// # Safety
///
/// - `model` must be NULL or a pointer previously returned by `oxbitnet_load`
///   that has not yet been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxbitnet_free(model: *mut OxBitNet) {
    if model.is_null() {
        return;
    }
    let mut handle = unsafe { Box::from_raw(model) };
    if let Some(ref mut m) = handle.inner {
        m.dispose();
    }
    // Box drops here, runtime shuts down
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

/// Generate text from a raw prompt string.
///
/// Tokens are delivered via `callback`. Return 0 from the callback to continue,
/// non-zero to stop early.
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
///
/// - `model` must be a valid pointer from `oxbitnet_load`.
/// - `prompt` must be a valid null-terminated UTF-8 string.
/// - `options` may be NULL for defaults.
/// - `callback` must be a valid function pointer or NULL (NULL = generate silently).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxbitnet_generate(
    model: *mut OxBitNet,
    prompt: *const c_char,
    options: *const OxBitNetGenerateOptions,
    callback: OxBitNetTokenFn,
    userdata: *mut std::ffi::c_void,
) -> i32 {
    clear_error();

    if model.is_null() {
        set_error("model must not be NULL");
        return -1;
    }
    if prompt.is_null() {
        set_error("prompt must not be NULL");
        return -1;
    }

    let handle = unsafe { &mut *model };
    let prompt_str = match unsafe { CStr::from_ptr(prompt) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(format!("invalid UTF-8 in prompt: {e}"));
            return -1;
        }
    };

    let gen_opts = read_generate_options(options);

    let bitnet = match handle.inner.as_mut() {
        Some(m) => m,
        None => {
            set_error("model has been disposed");
            return -1;
        }
    };

    let mut stream = bitnet.generate(prompt_str, gen_opts);
    drain_stream(&handle.runtime, &mut stream, callback, userdata);

    0
}

/// Generate text from chat messages.
///
/// Tokens are delivered via `callback`. Return 0 from the callback to continue,
/// non-zero to stop early.
///
/// Returns 0 on success, -1 on error.
///
/// # Safety
///
/// - `model` must be a valid pointer from `oxbitnet_load`.
/// - `messages` must point to `num_messages` valid `OxBitNetChatMessage` structs.
/// - `options` may be NULL for defaults.
/// - `callback` must be a valid function pointer or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxbitnet_chat(
    model: *mut OxBitNet,
    messages: *const OxBitNetChatMessage,
    num_messages: usize,
    options: *const OxBitNetGenerateOptions,
    callback: OxBitNetTokenFn,
    userdata: *mut std::ffi::c_void,
) -> i32 {
    clear_error();

    if model.is_null() {
        set_error("model must not be NULL");
        return -1;
    }
    if messages.is_null() && num_messages > 0 {
        set_error("messages must not be NULL when num_messages > 0");
        return -1;
    }

    let handle = unsafe { &mut *model };

    // Convert C messages to Rust
    let mut chat_messages = Vec::with_capacity(num_messages);
    for i in 0..num_messages {
        let msg = unsafe { &*messages.add(i) };

        let role = if msg.role.is_null() {
            set_error(format!("messages[{i}].role must not be NULL"));
            return -1;
        } else {
            match unsafe { CStr::from_ptr(msg.role) }.to_str() {
                Ok(s) => s.to_string(),
                Err(e) => {
                    set_error(format!("invalid UTF-8 in messages[{i}].role: {e}"));
                    return -1;
                }
            }
        };

        let content = if msg.content.is_null() {
            set_error(format!("messages[{i}].content must not be NULL"));
            return -1;
        } else {
            match unsafe { CStr::from_ptr(msg.content) }.to_str() {
                Ok(s) => s.to_string(),
                Err(e) => {
                    set_error(format!("invalid UTF-8 in messages[{i}].content: {e}"));
                    return -1;
                }
            }
        };

        chat_messages.push(oxbitnet::ChatMessage { role, content });
    }

    let gen_opts = read_generate_options(options);

    let bitnet = match handle.inner.as_mut() {
        Some(m) => m,
        None => {
            set_error("model has been disposed");
            return -1;
        }
    };

    let mut stream = bitnet.generate_chat(&chat_messages, gen_opts);
    drain_stream(&handle.runtime, &mut stream, callback, userdata);

    0
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

/// Get the last error message, or NULL if no error occurred.
///
/// The returned pointer is owned by the library and valid until the next
/// FFI call on the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn oxbitnet_error_message() -> *const c_char {
    LAST_ERROR.with(|cell| {
        let borrow = cell.borrow();
        match borrow.as_ref() {
            Some(s) => s.as_ptr(),
            None => ptr::null(),
        }
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Consume a token stream, forwarding each token to the C callback.
///
/// Blocks the current thread on the provided runtime until the stream is
/// exhausted or the callback returns non-zero.
fn drain_stream(
    runtime: &tokio::runtime::Runtime,
    stream: &mut (impl futures::Stream<Item = String> + Unpin),
    callback: OxBitNetTokenFn,
    userdata: *mut std::ffi::c_void,
) {
    runtime.block_on(async {
        while let Some(token) = stream.next().await {
            if let Some(cb) = callback {
                let c_str = match CString::new(token) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let len = c_str.as_bytes().len();
                let ret = unsafe { cb(c_str.as_ptr(), len, userdata) };
                if ret != 0 {
                    break;
                }
            }
        }
    });
}

/// Read generate options from a C pointer, falling back to defaults if NULL.
///
/// # Safety
///
/// `options` must be NULL or point to a valid `OxBitNetGenerateOptions`.
unsafe fn read_generate_options(options: *const OxBitNetGenerateOptions) -> oxbitnet::GenerateOptions {
    if options.is_null() {
        oxbitnet::GenerateOptions {
            repeat_penalty: 1.1,
            ..Default::default()
        }
    } else {
        let opts = unsafe { &*options };
        oxbitnet::GenerateOptions {
            max_tokens: opts.max_tokens,
            temperature: opts.temperature,
            top_k: opts.top_k,
            repeat_penalty: opts.repeat_penalty,
            repeat_last_n: opts.repeat_last_n,
        }
    }
}
