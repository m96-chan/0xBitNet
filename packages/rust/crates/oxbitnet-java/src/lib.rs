//! Java/JNI bindings for oxbitnet.
//!
//! Provides JNI native methods called by the `io.github.m96chan.oxbitnet.BitNet`
//! Java class. Follows the same pattern as the Python (PyO3) bindings: owns a
//! `tokio::Runtime`, calls `block_on()` to bridge async→sync, and delivers
//! tokens via callback.

use std::path::PathBuf;

use futures::StreamExt;
use jni::objects::{JClass, JObject, JString, JValue};
use jni::sys::{jfloat, jint, jlong};
use jni::JNIEnv;

struct JavaBitNet {
    inner: Option<oxbitnet::BitNet>,
    runtime: tokio::runtime::Runtime,
}

/// Throw an `OxBitNetException` and return `default`.
fn throw<T>(env: &mut JNIEnv, msg: &str, default: T) -> T {
    let _ = env.throw_new("io/github/m96chan/oxbitnet/OxBitNetException", msg);
    default
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

/// `static native long nativeLoad(String source, Object loadOptions);`
#[no_mangle]
pub extern "system" fn Java_io_github_m96chan_oxbitnet_BitNet_nativeLoad<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    source: JString<'local>,
    load_options: JObject<'local>,
) -> jlong {
    let source_str: String = match env.get_string(&source) {
        Ok(s) => s.into(),
        Err(e) => return throw(&mut env, &format!("Invalid source string: {e}"), 0),
    };

    let mut opts = oxbitnet::LoadOptions::default();

    // Extract fields from LoadOptions if non-null
    if !load_options.is_null() {
        // cacheDir
        if let Ok(cache_dir_obj) = env.get_field(&load_options, "cacheDir", "Ljava/lang/String;") {
            if let Ok(cache_dir_jobj) = cache_dir_obj.l() {
                if !cache_dir_jobj.is_null() {
                    let cache_dir_jstr = JString::from(cache_dir_jobj);
                    let cache_dir: Option<String> = env.get_string(&cache_dir_jstr).ok().map(|s| s.into());
                    if let Some(dir) = cache_dir {
                        opts.cache_dir = Some(PathBuf::from(dir));
                    }
                }
            }
        }

        // Progress callback — we need to call it from the async block.
        // We create a GlobalRef so it survives across the block_on boundary.
        if let Ok(progress_cb_val) =
            env.get_field(&load_options, "onProgress", "Lio/github/m96chan/oxbitnet/LoadOptions$ProgressCallback;")
        {
            if let Ok(progress_cb_obj) = progress_cb_val.l() {
                if !progress_cb_obj.is_null() {
                    if let Ok(global_cb) = env.new_global_ref(progress_cb_obj) {
                        // Get JavaVM so we can attach from the callback
                        if let Ok(jvm) = env.get_java_vm() {
                            opts.on_progress = Some(Box::new(move |p: oxbitnet::LoadProgress| {
                                if let Ok(mut cb_env) = jvm.attach_current_thread() {
                                    let phase_str = match p.phase {
                                        oxbitnet::model::loader::LoadPhase::Download => "download",
                                        oxbitnet::model::loader::LoadPhase::Parse => "parse",
                                        oxbitnet::model::loader::LoadPhase::Upload => "upload",
                                    };
                                    if let Ok(j_phase) = cb_env.new_string(phase_str) {
                                        let _ = cb_env.call_method(
                                            &global_cb,
                                            "onProgress",
                                            "(Ljava/lang/String;JJD)V",
                                            &[
                                                JValue::Object(&j_phase.into()),
                                                JValue::Long(p.loaded as i64),
                                                JValue::Long(p.total as i64),
                                                JValue::Double(p.fraction),
                                            ],
                                        );
                                    }
                                }
                            }));
                        }
                    }
                }
            }
        }
    }

    let runtime = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => return throw(&mut env, &format!("Failed to create runtime: {e}"), 0),
    };

    match runtime.block_on(oxbitnet::BitNet::load(&source_str, opts)) {
        Ok(bitnet) => {
            let handle = Box::new(JavaBitNet {
                inner: Some(bitnet),
                runtime,
            });
            Box::into_raw(handle) as jlong
        }
        Err(e) => throw(&mut env, &format!("{e}"), 0),
    }
}

// ---------------------------------------------------------------------------
// Generate (raw prompt)
// ---------------------------------------------------------------------------

/// `static native int nativeGenerate(long handle, String prompt, long maxTokens,
///     float temperature, long topK, float repeatPenalty, long repeatLastN,
///     TokenCallback callback);`
#[no_mangle]
pub extern "system" fn Java_io_github_m96chan_oxbitnet_BitNet_nativeGenerate<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    prompt: JString<'local>,
    max_tokens: jlong,
    temperature: jfloat,
    top_k: jlong,
    repeat_penalty: jfloat,
    repeat_last_n: jlong,
    callback: JObject<'local>,
) -> jint {
    if handle == 0 {
        return throw(&mut env, "Model handle is null", -1);
    }

    let prompt_str: String = match env.get_string(&prompt) {
        Ok(s) => s.into(),
        Err(e) => return throw(&mut env, &format!("Invalid prompt string: {e}"), -1),
    };

    let java_bitnet = unsafe { &mut *(handle as *mut JavaBitNet) };
    let bitnet = match java_bitnet.inner.as_mut() {
        Some(m) => m,
        None => return throw(&mut env, "Model has been disposed", -1),
    };

    let options = oxbitnet::GenerateOptions {
        max_tokens: max_tokens as usize,
        temperature,
        top_k: top_k as usize,
        repeat_penalty,
        repeat_last_n: repeat_last_n as usize,
    };

    let mut stream = bitnet.generate(&prompt_str, options);
    let mut count: i32 = 0;

    let has_callback = !callback.is_null();

    java_bitnet.runtime.block_on(async {
        while let Some(token) = stream.next().await {
            count += 1;
            if has_callback {
                match env.new_string(&token) {
                    Ok(j_token) => {
                        let result = env.call_method(
                            &callback,
                            "onToken",
                            "(Ljava/lang/String;)Z",
                            &[JValue::Object(&j_token.into())],
                        );
                        match result {
                            Ok(ret) => {
                                if let Ok(cont) = ret.z() {
                                    if !cont {
                                        break;
                                    }
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    Err(_) => continue,
                }
            }
        }
    });

    count
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

/// `static native int nativeChat(long handle, String[] roles, String[] contents,
///     long maxTokens, float temperature, long topK, float repeatPenalty,
///     long repeatLastN, TokenCallback callback);`
#[no_mangle]
pub extern "system" fn Java_io_github_m96chan_oxbitnet_BitNet_nativeChat<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    handle: jlong,
    roles: JObject<'local>,    // String[]
    contents: JObject<'local>, // String[]
    max_tokens: jlong,
    temperature: jfloat,
    top_k: jlong,
    repeat_penalty: jfloat,
    repeat_last_n: jlong,
    callback: JObject<'local>,
) -> jint {
    use jni::objects::JObjectArray;

    if handle == 0 {
        return throw(&mut env, "Model handle is null", -1);
    }

    let roles_arr: &JObjectArray = roles.as_ref().into();
    let contents_arr: &JObjectArray = contents.as_ref().into();

    let num_messages = match env.get_array_length(roles_arr) {
        Ok(n) => n as usize,
        Err(e) => return throw(&mut env, &format!("Failed to get roles array length: {e}"), -1),
    };

    let mut chat_messages = Vec::with_capacity(num_messages);
    for i in 0..num_messages {
        let role_obj = match env.get_object_array_element(roles_arr, i as i32) {
            Ok(o) => o,
            Err(e) => {
                return throw(&mut env, &format!("Failed to get role[{i}]: {e}"), -1);
            }
        };
        let content_obj = match env.get_object_array_element(contents_arr, i as i32) {
            Ok(o) => o,
            Err(e) => {
                return throw(&mut env, &format!("Failed to get content[{i}]: {e}"), -1);
            }
        };

        let role: String = match env.get_string(&JString::from(role_obj)) {
            Ok(s) => s.into(),
            Err(e) => {
                return throw(&mut env, &format!("Invalid role string at [{i}]: {e}"), -1);
            }
        };
        let content: String = match env.get_string(&JString::from(content_obj)) {
            Ok(s) => s.into(),
            Err(e) => {
                return throw(&mut env, &format!("Invalid content string at [{i}]: {e}"), -1);
            }
        };

        chat_messages.push(oxbitnet::ChatMessage { role, content });
    }

    let java_bitnet = unsafe { &mut *(handle as *mut JavaBitNet) };
    let bitnet = match java_bitnet.inner.as_mut() {
        Some(m) => m,
        None => return throw(&mut env, "Model has been disposed", -1),
    };

    let options = oxbitnet::GenerateOptions {
        max_tokens: max_tokens as usize,
        temperature,
        top_k: top_k as usize,
        repeat_penalty,
        repeat_last_n: repeat_last_n as usize,
    };

    let mut stream = bitnet.generate_chat(&chat_messages, options);
    let mut count: i32 = 0;

    let has_callback = !callback.is_null();

    java_bitnet.runtime.block_on(async {
        while let Some(token) = stream.next().await {
            count += 1;
            if has_callback {
                match env.new_string(&token) {
                    Ok(j_token) => {
                        let result = env.call_method(
                            &callback,
                            "onToken",
                            "(Ljava/lang/String;)Z",
                            &[JValue::Object(&j_token.into())],
                        );
                        match result {
                            Ok(ret) => {
                                if let Ok(cont) = ret.z() {
                                    if !cont {
                                        break;
                                    }
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    Err(_) => continue,
                }
            }
        }
    });

    count
}

// ---------------------------------------------------------------------------
// Free
// ---------------------------------------------------------------------------

/// `static native void nativeFree(long handle);`
#[no_mangle]
pub extern "system" fn Java_io_github_m96chan_oxbitnet_BitNet_nativeFree(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle == 0 {
        return;
    }
    let mut java_bitnet = unsafe { Box::from_raw(handle as *mut JavaBitNet) };
    if let Some(ref mut model) = java_bitnet.inner {
        model.dispose();
    }
    // Box drops here, runtime shuts down
}
