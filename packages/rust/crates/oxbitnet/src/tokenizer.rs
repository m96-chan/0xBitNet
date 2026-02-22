use crate::error::{BitNetError, Result};
use crate::model::gguf::GgufMetadata;

/// Wraps the HuggingFace `tokenizers` crate for BPE tokenization.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    bos_id: u32,
    eos_id: u32,
    eot_id: Option<u32>,
    im_end_id: Option<u32>,
}

/// Chat message for apply_chat_template.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl Tokenizer {
    /// Create a tokenizer from GGUF metadata (extracts vocab + merges).
    pub fn from_gguf_metadata(metadata: &GgufMetadata) -> Result<Self> {
        let tokens = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_string_array())
            .ok_or_else(|| BitNetError::Tokenizer("Missing tokenizer.ggml.tokens".into()))?;

        let merges_raw = metadata
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_string_array());

        let bos_id = metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(1);
        let eos_id = metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(2);

        // Build tokenizer from vocab and merges
        let vocab: ahash::AHashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i as u32))
            .collect();

        let merges: Vec<(String, String)> = merges_raw
            .unwrap_or_default()
            .iter()
            .filter_map(|m| {
                let parts: Vec<&str> = m.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();

        let bpe = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab, merges)
            .unk_token("<unk>".to_string())
            .build()
            .map_err(|e| BitNetError::Tokenizer(format!("Failed to build BPE: {e}")))?;

        let mut tokenizer = tokenizers::Tokenizer::new(bpe);

        // GPT-2 style pre-tokenizer
        tokenizer.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, false),
        ));

        // Byte-level decoder
        tokenizer.with_decoder(Some(
            tokenizers::decoders::byte_level::ByteLevel::new(false, true, false),
        ));

        // Find special stop tokens
        let eot_id = tokenizer.token_to_id("<|eot_id|>");
        let im_end_id = tokenizer.token_to_id("<|im_end|>");

        Ok(Self {
            inner: tokenizer,
            bos_id,
            eos_id,
            eot_id,
            im_end_id,
        })
    }

    /// Create a tokenizer from a tokenizer.json file.
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| BitNetError::Tokenizer(format!("Failed to load tokenizer: {e}")))?;

        Ok(Self {
            eot_id: tokenizer.token_to_id("<|eot_id|>"),
            im_end_id: tokenizer.token_to_id("<|im_end|>"),
            inner: tokenizer,
            bos_id: 1,
            eos_id: 2,
        })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| BitNetError::Tokenizer(format!("Encode failed: {e}")))?;

        let mut ids: Vec<u32> = Vec::new();
        if add_bos {
            ids.push(self.bos_id);
        }
        ids.extend_from_slice(encoding.get_ids());
        Ok(ids)
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        // Filter out BOS/EOS
        let filtered: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&id| id != self.bos_id && id != self.eos_id)
            .collect();

        self.inner
            .decode(&filtered, true)
            .map_err(|e| BitNetError::Tokenizer(format!("Decode failed: {e}")))
    }

    /// Decode a single token ID.
    pub fn decode_one(&self, id: u32) -> Result<String> {
        self.inner
            .decode(&[id], true)
            .map_err(|e| BitNetError::Tokenizer(format!("Decode failed: {e}")))
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_id
    }

    pub fn bos_token_id(&self) -> u32 {
        self.bos_id
    }

    pub fn eot_token_id(&self) -> Option<u32> {
        self.eot_id
    }

    pub fn im_end_token_id(&self) -> Option<u32> {
        self.im_end_id
    }

    /// Apply the appropriate chat template to a list of messages.
    /// Auto-detects ChatML vs LLaMA 3 format from vocab.
    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        // Check for ChatML tokens first
        let im_start = self.inner.token_to_id("<|im_start|>");
        let im_end = self.inner.token_to_id("<|im_end|>");
        if let (Some(im_start), Some(im_end)) = (im_start, im_end) {
            return self.apply_chatml(messages, im_start, im_end);
        }

        // Check for LLaMA 3 tokens
        let start_header = self.inner.token_to_id("<|start_header_id|>");
        let end_header = self.inner.token_to_id("<|end_header_id|>");
        let eot = self.inner.token_to_id("<|eot_id|>");

        // Fallback to plain encoding if special tokens missing
        if start_header.is_none() || end_header.is_none() || eot.is_none() {
            let text: String = messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n");
            return self.encode(&text, true);
        }

        let start_header = start_header.unwrap();
        let end_header = end_header.unwrap();
        let eot = eot.unwrap();

        let mut tokens = vec![self.bos_id];

        for msg in messages {
            tokens.push(start_header);
            tokens.extend(self.encode(&msg.role, false)?);
            tokens.push(end_header);
            tokens.extend(self.encode(&format!("\n\n{}", msg.content), false)?);
            tokens.push(eot);
        }

        // Trailing assistant header
        tokens.push(start_header);
        tokens.extend(self.encode("assistant", false)?);
        tokens.push(end_header);
        tokens.extend(self.encode("\n\n", false)?);

        Ok(tokens)
    }

    /// Apply the ChatML template (used by Falcon-E and similar models).
    /// Format: <|im_start|>role\ncontent<|im_end|>\n
    fn apply_chatml(&self, messages: &[ChatMessage], im_start: u32, im_end: u32) -> Result<Vec<u32>> {
        let mut tokens = vec![self.bos_id];
        for msg in messages {
            tokens.push(im_start);
            tokens.extend(self.encode(&format!("{}\n{}", msg.role, msg.content), false)?);
            tokens.push(im_end);
            tokens.extend(self.encode("\n", false)?);
        }
        // Trailing assistant prompt
        tokens.push(im_start);
        tokens.extend(self.encode("assistant\n", false)?);
        Ok(tokens)
    }
}
