use thiserror::Error;

#[derive(Error, Debug)]
pub enum BitNetError {
    #[error("GPU error: {0}")]
    Gpu(String),

    #[error("Failed to request GPU adapter")]
    NoAdapter,

    #[error("Failed to request GPU device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),

    #[error("Invalid GGUF magic: 0x{0:08x} (expected 0x46554747)")]
    InvalidGgufMagic(u32),

    #[error("Unsupported GGUF version: {0}")]
    UnsupportedGgufVersion(u32),

    #[error("Unsupported GGML type: {0}")]
    UnsupportedGgmlType(u32),

    #[error("Unknown GGUF metadata type: {0}")]
    UnknownMetadataType(u32),

    #[error("Missing weight tensor: \"{0}\"")]
    MissingWeight(String),

    #[error("GGUF parse error: {0}")]
    GgufParse(String),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Buffer mapping failed")]
    BufferMap,

    #[error("Model not loaded")]
    NotLoaded,

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, BitNetError>;
