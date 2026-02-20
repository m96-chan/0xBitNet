"""Type stubs for oxbitnet."""

class BitNet:
    """BitNet b1.58 ternary LLM inference engine backed by wgpu."""

    @staticmethod
    def load_sync(source: str) -> "BitNet":
        """Load a model from a URL or local GGUF file path."""
        ...

    def generate_sync(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str:
        """Generate text synchronously, returning the full output."""
        ...

    def generate_tokens_sync(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> list[str]:
        """Generate text as a list of token strings."""
        ...

    def dispose(self) -> None:
        """Release all GPU resources."""
        ...
