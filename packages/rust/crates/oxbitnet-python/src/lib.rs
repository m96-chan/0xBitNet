use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

/// Python wrapper for the BitNet model.
#[pyclass]
struct BitNet {
    inner: Option<oxbitnet::BitNet>,
    runtime: tokio::runtime::Runtime,
}

fn make_options(
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    repeat_penalty: f32,
) -> oxbitnet::GenerateOptions {
    oxbitnet::GenerateOptions {
        max_tokens,
        temperature,
        top_k,
        repeat_penalty,
        ..Default::default()
    }
}

#[pymethods]
impl BitNet {
    /// Load a BitNet model from a URL or local path.
    #[staticmethod]
    #[pyo3(name = "load_sync")]
    fn load_sync(source: &str) -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {e}")))?;

        let inner = runtime
            .block_on(oxbitnet::BitNet::load(source, Default::default()))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

        Ok(Self {
            inner: Some(inner),
            runtime,
        })
    }

    /// Generate text with streaming via callback (raw prompt, no chat template).
    /// Returns total number of tokens generated.
    #[pyo3(
        name = "generate",
        signature = (prompt, on_token, max_tokens=256, temperature=1.0, top_k=50, repeat_penalty=1.1)
    )]
    fn generate(
        &mut self,
        py: Python<'_>,
        prompt: &str,
        on_token: Py<PyAny>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        repeat_penalty: f32,
    ) -> PyResult<usize> {
        use futures::StreamExt;

        let model = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Model disposed"))?;

        let options = make_options(max_tokens, temperature, top_k, repeat_penalty);
        let mut stream = model.generate(prompt, options);
        let mut count = 0usize;

        let on_token_ref = &on_token;
        self.runtime.block_on(async {
            while let Some(token) = stream.next().await {
                count += 1;
                py.check_signals().ok();
                let _ = on_token_ref.call1(py, (token,));
            }
        });

        Ok(count)
    }

    /// Generate text with chat template and streaming via callback.
    ///
    /// messages: list of dicts with "role" and "content" keys.
    /// Returns total number of tokens generated.
    #[pyo3(
        name = "chat",
        signature = (messages, on_token, max_tokens=256, temperature=1.0, top_k=50, repeat_penalty=1.1)
    )]
    fn chat(
        &mut self,
        py: Python<'_>,
        messages: Vec<(String, String)>,
        on_token: Py<PyAny>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        repeat_penalty: f32,
    ) -> PyResult<usize> {
        use futures::StreamExt;

        let model = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Model disposed"))?;

        let chat_messages: Vec<oxbitnet::ChatMessage> = messages
            .into_iter()
            .map(|(role, content)| oxbitnet::ChatMessage { role, content })
            .collect();

        let options = make_options(max_tokens, temperature, top_k, repeat_penalty);
        let mut stream = model.generate_chat(&chat_messages, options);
        let mut count = 0usize;

        let on_token_ref = &on_token;
        self.runtime.block_on(async {
            while let Some(token) = stream.next().await {
                count += 1;
                py.check_signals().ok();
                let _ = on_token_ref.call1(py, (token,));
            }
        });

        Ok(count)
    }

    /// Generate text synchronously (raw prompt), returning the full output string.
    #[pyo3(
        name = "generate_sync",
        signature = (prompt, max_tokens=256, temperature=1.0, top_k=50, repeat_penalty=1.1)
    )]
    fn generate_sync(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        repeat_penalty: f32,
    ) -> PyResult<String> {
        use futures::StreamExt;

        let model = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Model disposed"))?;

        let options = make_options(max_tokens, temperature, top_k, repeat_penalty);
        let mut stream = model.generate(prompt, options);
        let mut output = String::new();

        self.runtime.block_on(async {
            while let Some(token) = stream.next().await {
                output.push_str(&token);
            }
        });

        Ok(output)
    }

    /// Generate text as a list of token strings.
    #[pyo3(
        name = "generate_tokens_sync",
        signature = (prompt, max_tokens=256, temperature=1.0, top_k=50, repeat_penalty=1.1)
    )]
    fn generate_tokens_sync(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        repeat_penalty: f32,
    ) -> PyResult<Vec<String>> {
        use futures::StreamExt;

        let model = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("Model disposed"))?;

        let options = make_options(max_tokens, temperature, top_k, repeat_penalty);
        let mut stream = model.generate(prompt, options);
        let mut tokens = Vec::new();

        self.runtime.block_on(async {
            while let Some(token) = stream.next().await {
                tokens.push(token);
            }
        });

        Ok(tokens)
    }

    /// Release all GPU resources.
    fn dispose(&mut self) {
        if let Some(ref mut model) = self.inner {
            model.dispose();
        }
        self.inner = None;
    }

    fn __repr__(&self) -> String {
        if self.inner.is_some() {
            "BitNet(loaded)".to_string()
        } else {
            "BitNet(disposed)".to_string()
        }
    }
}

/// oxbitnet — Run BitNet b1.58 ternary LLMs with wgpu
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BitNet>()?;
    Ok(())
}
