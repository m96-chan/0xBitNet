use std::collections::HashMap;

use crate::error::{BitNetError, Result};

// GGUF magic: "GGUF" as little-endian uint32
const GGUF_MAGIC: u32 = 0x4655_4747;

// GGUF metadata value type enum
const META_UINT8: u32 = 0;
const META_INT8: u32 = 1;
const META_UINT16: u32 = 2;
const META_INT16: u32 = 3;
const META_UINT32: u32 = 4;
const META_INT32: u32 = 5;
const META_FLOAT32: u32 = 6;
const META_BOOL: u32 = 7;
const META_STRING: u32 = 8;
const META_ARRAY: u32 = 9;
const META_UINT64: u32 = 10;
const META_INT64: u32 = 11;
const META_FLOAT64: u32 = 12;

// GGML quantization types
pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_I8: u32 = 16;
pub const GGML_TYPE_I16: u32 = 17;
pub const GGML_TYPE_I32: u32 = 18;
pub const GGML_TYPE_I64: u32 = 27;
pub const GGML_TYPE_F64: u32 = 28;
pub const GGML_TYPE_BF16: u32 = 30;
pub const GGML_TYPE_TQ1_0: u32 = 34;
pub const GGML_TYPE_TQ2_0: u32 = 35;
/// Ternary {-1, 0, 1} packed 2-bit (4 values/byte)
pub const GGML_TYPE_I2_S: u32 = 36;

/// Bytes per element for GGML types (fractional for sub-byte).
pub fn ggml_type_size(ty: u32) -> Result<f64> {
    match ty {
        GGML_TYPE_F32 => Ok(4.0),
        GGML_TYPE_F16 => Ok(2.0),
        GGML_TYPE_I8 => Ok(1.0),
        GGML_TYPE_I16 => Ok(2.0),
        GGML_TYPE_I32 => Ok(4.0),
        GGML_TYPE_I64 => Ok(8.0),
        GGML_TYPE_F64 => Ok(8.0),
        GGML_TYPE_BF16 => Ok(2.0),
        GGML_TYPE_TQ1_0 => Ok(54.0 / 256.0),
        GGML_TYPE_TQ2_0 => Ok(66.0 / 256.0),
        GGML_TYPE_I2_S => Ok(0.25),
        _ => Err(BitNetError::UnsupportedGgmlType(ty)),
    }
}

/// Metadata value from a GGUF file.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            GgufValue::I32(v) => Some(*v as u32),
            GgufValue::U8(v) => Some(*v as u32),
            GgufValue::U16(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        match self {
            GgufValue::Array(arr) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    result.push(v.as_str()?);
                }
                Some(result)
            }
            _ => None,
        }
    }
}

pub type GgufMetadata = HashMap<String, GgufValue>;

#[derive(Debug)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dimensions: u32,
    pub shape: Vec<u64>,
    pub tensor_type: u32,
    pub offset: u64,
}

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: GgufMetadata,
    pub tensors: Vec<GgufTensorInfo>,
    pub tensor_data_offset: usize,
}

/// Streaming GGUF parser.
pub struct GgufParser<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> GgufParser<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    pub fn parse(&mut self) -> Result<GgufFile> {
        // Header
        let magic = self.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(BitNetError::InvalidGgufMagic(magic));
        }
        let version = self.read_u32()?;
        if !(2..=3).contains(&version) {
            return Err(BitNetError::UnsupportedGgufVersion(version));
        }
        let tensor_count = self.read_u64()?;
        let metadata_kv_count = self.read_u64()?;

        // Metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = self.read_string()?;
            let value = self.read_metadata_value()?;
            metadata.insert(key, value);
        }

        // Tensor infos
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = self.read_string()?;
            let n_dimensions = self.read_u32()?;
            let mut shape = Vec::with_capacity(n_dimensions as usize);
            for _ in 0..n_dimensions {
                shape.push(self.read_u64()?);
            }
            let tensor_type = self.read_u32()?;
            let offset = self.read_u64()?;
            tensors.push(GgufTensorInfo {
                name,
                n_dimensions,
                shape,
                tensor_type,
                offset,
            });
        }

        // Tensor data starts after alignment padding
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as usize;
        let tensor_data_offset = self.offset.div_ceil(alignment) * alignment;

        Ok(GgufFile {
            version,
            tensor_count,
            metadata,
            tensors,
            tensor_data_offset,
        })
    }

    fn read_metadata_value(&mut self) -> Result<GgufValue> {
        let value_type = self.read_u32()?;
        self.read_value_of_type(value_type)
    }

    fn read_value_of_type(&mut self, ty: u32) -> Result<GgufValue> {
        match ty {
            META_UINT8 => Ok(GgufValue::U8(self.read_u8()?)),
            META_INT8 => Ok(GgufValue::I8(self.read_i8()?)),
            META_UINT16 => Ok(GgufValue::U16(self.read_u16()?)),
            META_INT16 => Ok(GgufValue::I16(self.read_i16()?)),
            META_UINT32 => Ok(GgufValue::U32(self.read_u32()?)),
            META_INT32 => Ok(GgufValue::I32(self.read_i32()?)),
            META_FLOAT32 => Ok(GgufValue::F32(self.read_f32()?)),
            META_BOOL => Ok(GgufValue::Bool(self.read_u8()? != 0)),
            META_STRING => Ok(GgufValue::String(self.read_string()?)),
            META_UINT64 => Ok(GgufValue::U64(self.read_u64()?)),
            META_INT64 => Ok(GgufValue::I64(self.read_i64()?)),
            META_FLOAT64 => Ok(GgufValue::F64(self.read_f64()?)),
            META_ARRAY => {
                let elem_type = self.read_u32()?;
                let len = self.read_u64()? as usize;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(self.read_value_of_type(elem_type)?);
                }
                Ok(GgufValue::Array(arr))
            }
            _ => Err(BitNetError::UnknownMetadataType(ty)),
        }
    }

    // --- Primitive readers ---

    fn ensure(&self, n: usize) -> Result<()> {
        if self.offset + n > self.data.len() {
            Err(BitNetError::GgufParse(format!(
                "Unexpected EOF at offset {} (need {} bytes, have {})",
                self.offset,
                n,
                self.data.len()
            )))
        } else {
            Ok(())
        }
    }

    fn read_u8(&mut self) -> Result<u8> {
        self.ensure(1)?;
        let v = self.data[self.offset];
        self.offset += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        self.ensure(2)?;
        let v = u16::from_le_bytes(self.data[self.offset..self.offset + 2].try_into().unwrap());
        self.offset += 2;
        Ok(v)
    }

    fn read_i16(&mut self) -> Result<i16> {
        self.ensure(2)?;
        let v = i16::from_le_bytes(self.data[self.offset..self.offset + 2].try_into().unwrap());
        self.offset += 2;
        Ok(v)
    }

    fn read_u32(&mut self) -> Result<u32> {
        self.ensure(4)?;
        let v = u32::from_le_bytes(self.data[self.offset..self.offset + 4].try_into().unwrap());
        self.offset += 4;
        Ok(v)
    }

    fn read_i32(&mut self) -> Result<i32> {
        self.ensure(4)?;
        let v = i32::from_le_bytes(self.data[self.offset..self.offset + 4].try_into().unwrap());
        self.offset += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> Result<u64> {
        self.ensure(8)?;
        let v = u64::from_le_bytes(self.data[self.offset..self.offset + 8].try_into().unwrap());
        self.offset += 8;
        Ok(v)
    }

    fn read_i64(&mut self) -> Result<i64> {
        self.ensure(8)?;
        let v = i64::from_le_bytes(self.data[self.offset..self.offset + 8].try_into().unwrap());
        self.offset += 8;
        Ok(v)
    }

    fn read_f32(&mut self) -> Result<f32> {
        self.ensure(4)?;
        let v = f32::from_le_bytes(self.data[self.offset..self.offset + 4].try_into().unwrap());
        self.offset += 4;
        Ok(v)
    }

    fn read_f64(&mut self) -> Result<f64> {
        self.ensure(8)?;
        let v = f64::from_le_bytes(self.data[self.offset..self.offset + 8].try_into().unwrap());
        self.offset += 8;
        Ok(v)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        self.ensure(len)?;
        let s = std::str::from_utf8(&self.data[self.offset..self.offset + len])
            .map_err(|e| BitNetError::GgufParse(format!("Invalid UTF-8 in string: {e}")))?;
        self.offset += len;
        Ok(s.to_string())
    }
}
