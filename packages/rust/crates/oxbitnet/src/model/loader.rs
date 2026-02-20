use std::path::PathBuf;
use std::sync::Arc;

use half::f16;
use sha2::{Digest, Sha256};
use tracing::{debug, info};

use crate::error::Result;
use crate::model::config::{Activation, ModelConfig};
use crate::model::gguf::{
    self, GgufMetadata, GgufParser, GgufValue, GGML_TYPE_F16, GGML_TYPE_I2_S,
};
use crate::model::weights::WeightStore;

#[derive(Debug, Clone, Copy)]
pub enum LoadPhase {
    Download,
    Parse,
    Upload,
}

#[derive(Debug, Clone)]
pub struct LoadProgress {
    pub phase: LoadPhase,
    pub loaded: u64,
    pub total: u64,
    pub fraction: f64,
}

#[derive(Default)]
pub struct LoadOptions {
    pub on_progress: Option<Box<dyn Fn(LoadProgress) + Send>>,
    pub cache_dir: Option<PathBuf>,
}


pub struct LoadResult {
    pub config: ModelConfig,
    pub weights: WeightStore,
    pub metadata: Option<GgufMetadata>,
}

/// Load a model from a URL or local path.
pub async fn load_model(
    source: &str,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    options: LoadOptions,
) -> Result<LoadResult> {
    let progress = |p: LoadProgress| {
        if let Some(ref cb) = options.on_progress {
            cb(p);
        }
    };

    progress(LoadProgress {
        phase: LoadPhase::Download,
        loaded: 0,
        total: 0,
        fraction: 0.0,
    });

    let data = fetch_model(source, &options, &progress).await?;

    progress(LoadProgress {
        phase: LoadPhase::Parse,
        loaded: 0,
        total: 1,
        fraction: 0.0,
    });

    load_gguf(&data, device, queue, &progress)
}

fn load_gguf(
    data: &[u8],
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    progress: &dyn Fn(LoadProgress),
) -> Result<LoadResult> {
    let mut parser = GgufParser::new(data);
    let gguf = parser.parse()?;

    let mut config = config_from_gguf_metadata(&gguf.metadata);

    // Detect tied embeddings
    let has_output_weight = gguf.tensors.iter().any(|t| t.name == "output.weight");
    config.tie_word_embeddings = !has_output_weight;

    debug!(
        "config: heads={}, kv_heads={}, head_dim={}, hidden={}, intermediate={}, layers={}, tied={}",
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim(),
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.tie_word_embeddings,
    );

    let mut store = WeightStore::new(Arc::clone(&device), Arc::clone(&queue));
    let max_binding = device.limits().max_storage_buffer_binding_size;
    let total_tensors = gguf.tensors.len();

    for (i, tensor) in gguf.tensors.iter().enumerate() {
        let data_offset = gguf.tensor_data_offset + tensor.offset as usize;

        let num_elements: u64 = tensor.shape.iter().product();
        let byte_size = if tensor.tensor_type == GGML_TYPE_I2_S {
            num_elements.div_ceil(4) as usize + 32
        } else {
            let elem_size = gguf::ggml_type_size(tensor.tensor_type)?;
            (num_elements as f64 * elem_size).ceil() as usize
        };

        let tensor_data = &data[data_offset..data_offset + byte_size];
        let hf_name = remap_gguf_name(&tensor.name);

        debug!(
            "tensor: {} -> {} (type={}, {} bytes)",
            tensor.name, hf_name, tensor.tensor_type, byte_size
        );

        if tensor.tensor_type == GGML_TYPE_I2_S {
            let packed_bytes = num_elements.div_ceil(4) as usize;
            let weight_data = &tensor_data[..packed_bytes];
            store.upload_sharded(&hf_name, weight_data, max_binding);

            // Extract per-tensor scale
            let scale_bytes = &tensor_data[packed_bytes..packed_bytes + 4];
            let tensor_scale = f32::from_le_bytes(scale_bytes.try_into().unwrap());
            let out_dim = tensor.shape.get(1).copied().unwrap_or(1) as usize;
            let scale_name = hf_name.replace(".weight", ".weight_scale");
            let scale_data: Vec<u8> = std::iter::repeat_n(tensor_scale.to_le_bytes(), out_dim)
                .flatten()
                .collect();
            store.upload(&scale_name, &scale_data);
        } else if tensor.tensor_type == GGML_TYPE_F16 {
            if hf_name == "model.embed_tokens.weight" {
                // Keep embedding as F16 on GPU
                store.upload_sharded(&hf_name, tensor_data, max_binding);
            } else {
                // Convert F16 to F32
                let f32_data = convert_f16_to_f32(tensor_data, num_elements as usize);
                store.upload_sharded(&hf_name, &f32_data, max_binding);
            }
        } else {
            store.upload_sharded(&hf_name, tensor_data, max_binding);
        }

        progress(LoadProgress {
            phase: LoadPhase::Upload,
            loaded: (i + 1) as u64,
            total: total_tensors as u64,
            fraction: (i + 1) as f64 / total_tensors as f64,
        });
    }

    info!("{} tensors loaded", total_tensors);

    create_dummy_scales(&mut store, &config);

    Ok(LoadResult {
        config,
        weights: store,
        metadata: Some(gguf.metadata),
    })
}

fn convert_f16_to_f32(src: &[u8], num_elements: usize) -> Vec<u8> {
    let mut dst = vec![0u8; num_elements * 4];
    for i in 0..num_elements {
        let h = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
        let f = f16::from_bits(h).to_f32();
        dst[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    dst
}

fn remap_gguf_name(name: &str) -> String {
    match name {
        "token_embd.weight" => return "model.embed_tokens.weight".to_string(),
        "output_norm.weight" => return "model.norm.weight".to_string(),
        "output.weight" => return "lm_head.weight".to_string(),
        _ => {}
    }

    // Block-level tensors: blk.{i}.{component}
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot_pos) = rest.find('.') {
            let layer = &rest[..dot_pos];
            let component = &rest[dot_pos + 1..];
            let prefix = format!("model.layers.{layer}");

            let mapped = match component {
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "attn_norm.weight" => "input_layernorm.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                "attn_sub_norm.weight" => "self_attn.sub_norm.weight",
                "ffn_sub_norm.weight" => "mlp.sub_norm.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                other => return format!("{prefix}.{other}"),
            };
            return format!("{prefix}.{mapped}");
        }
    }

    name.to_string()
}

fn config_from_gguf_metadata(metadata: &GgufMetadata) -> ModelConfig {
    let arch = metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("bitnet")
        .to_string();

    let get = |suffix: &str| -> Option<&GgufValue> {
        metadata
            .get(&format!("{arch}.{suffix}"))
            .or_else(|| metadata.get(&format!("llama.{suffix}")))
            .or_else(|| metadata.get(&format!("bitnet.{suffix}")))
            .or_else(|| metadata.get(&format!("bitnet-25.{suffix}")))
    };

    let hidden_size = get("embedding_length")
        .and_then(|v| v.as_u32())
        .unwrap_or(2560) as usize;
    let num_layers = get("block_count")
        .and_then(|v| v.as_u32())
        .unwrap_or(30) as usize;
    let num_heads = get("attention.head_count")
        .and_then(|v| v.as_u32())
        .unwrap_or(20) as usize;
    let num_kv_heads = get("attention.head_count_kv")
        .and_then(|v| v.as_u32())
        .unwrap_or(num_heads as u32) as usize;

    let vocab_size = get("vocab_size")
        .and_then(|v| v.as_u32())
        .map(|v| v as usize)
        .or_else(|| {
            metadata
                .get("tokenizer.ggml.tokens")
                .and_then(|v| v.as_string_array())
                .map(|a| a.len())
        })
        .unwrap_or(128256);

    let intermediate_size = get("feed_forward_length")
        .and_then(|v| v.as_u32())
        .unwrap_or(6912) as usize;

    let is_official = vocab_size > 100000 || arch.contains("bitnet");

    ModelConfig {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers: num_layers,
        num_attention_heads: num_heads,
        num_key_value_heads: num_kv_heads,
        max_position_embeddings: get("context_length")
            .and_then(|v| v.as_u32())
            .unwrap_or(4096) as usize,
        rms_norm_eps: get("attention.layer_norm_rms_epsilon")
            .and_then(|v| v.as_f32())
            .unwrap_or(1e-5),
        rope_theta: get("rope.freq_base")
            .and_then(|v| v.as_f32())
            .unwrap_or(if is_official { 500000.0 } else { 10000.0 }),
        tie_word_embeddings: false,
        activation: if is_official {
            Activation::Relu2
        } else {
            Activation::Silu
        },
    }
}

fn create_dummy_scales(store: &mut WeightStore, config: &ModelConfig) {
    let head_d = config.head_dim();

    for i in 0..config.num_hidden_layers {
        let p = format!("model.layers.{i}");
        let entries = [
            (
                format!("{p}.self_attn.q_proj.weight_scale"),
                config.num_attention_heads * head_d,
            ),
            (
                format!("{p}.self_attn.k_proj.weight_scale"),
                config.num_key_value_heads * head_d,
            ),
            (
                format!("{p}.self_attn.v_proj.weight_scale"),
                config.num_key_value_heads * head_d,
            ),
            (
                format!("{p}.self_attn.o_proj.weight_scale"),
                config.hidden_size,
            ),
            (
                format!("{p}.mlp.up_proj.weight_scale"),
                config.intermediate_size,
            ),
            (
                format!("{p}.mlp.down_proj.weight_scale"),
                config.hidden_size,
            ),
            (
                format!("{p}.mlp.gate_proj.weight_scale"),
                config.intermediate_size,
            ),
        ];

        for (name, dim) in entries {
            if !store.has(&name) {
                let data: Vec<u8> = std::iter::repeat_n(1.0f32.to_le_bytes(), dim)
                    .flatten()
                    .collect();
                store.upload(&name, &data);
            }
        }
    }

    let lm_head_scale = "lm_head.weight_scale".to_string();
    if !store.has(&lm_head_scale) {
        let data: Vec<u8> = std::iter::repeat_n(1.0f32.to_le_bytes(), config.vocab_size)
            .flatten()
            .collect();
        store.upload(&lm_head_scale, &data);
    }
}

/// Fetch model data from URL or local file, with disk caching.
async fn fetch_model(
    source: &str,
    options: &LoadOptions,
    progress: &dyn Fn(LoadProgress),
) -> Result<Vec<u8>> {
    // Local file
    if source.starts_with('/') || source.starts_with('.') || !source.contains("://") {
        let data = tokio::fs::read(source).await?;
        progress(LoadProgress {
            phase: LoadPhase::Download,
            loaded: data.len() as u64,
            total: data.len() as u64,
            fraction: 1.0,
        });
        return Ok(data);
    }

    // Check disk cache
    let cache_dir = options
        .cache_dir
        .clone()
        .or_else(|| dirs::cache_dir().map(|d| d.join(".0xbitnet")));

    if let Some(ref cache_dir) = cache_dir {
        let hash = format!("{:x}", Sha256::digest(source.as_bytes()));
        let cache_path = cache_dir.join(&hash);
        if cache_path.exists() {
            info!("Loading from cache: {}", cache_path.display());
            let data = tokio::fs::read(&cache_path).await?;
            progress(LoadProgress {
                phase: LoadPhase::Download,
                loaded: data.len() as u64,
                total: data.len() as u64,
                fraction: 1.0,
            });
            return Ok(data);
        }
    }

    // HTTP download
    info!("Downloading: {source}");
    let response = reqwest::get(source).await?;
    let total = response.content_length().unwrap_or(0);
    let bytes = response.bytes().await?;
    let data = bytes.to_vec();

    progress(LoadProgress {
        phase: LoadPhase::Download,
        loaded: data.len() as u64,
        total,
        fraction: 1.0,
    });

    // Save to cache
    if let Some(ref cache_dir) = cache_dir {
        let hash = format!("{:x}", Sha256::digest(source.as_bytes()));
        let cache_path = cache_dir.join(&hash);
        if let Err(e) = tokio::fs::create_dir_all(cache_dir).await {
            debug!("Failed to create cache dir: {e}");
        } else if let Err(e) = tokio::fs::write(&cache_path, &data).await {
            debug!("Failed to write cache: {e}");
        } else {
            info!("Cached to: {}", cache_path.display());
        }
    }

    Ok(data)
}
