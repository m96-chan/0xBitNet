import type { TokenizerConfig, ChatMessage } from "../types.js";

export type { ChatMessage };

/**
 * Tokenizer with support for:
 * - BPE (Byte Pair Encoding) — used by tiktoken-compatible models (2B-4T)
 * - SentencePiece BPE — used by community models
 *
 * Loads vocabulary and merge rules from a JSON blob (extracted from GGUF
 * metadata or downloaded separately).
 */
export class Tokenizer {
  private config: TokenizerConfig;
  private vocab: Map<string, number>;
  private reverseVocab: Map<number, string>;
  private merges: [string, string][];
  private mergeRanks: Map<string, number>;

  // Special tokens
  private bosId: number;
  private eosId: number;

  private textEncoder = new TextEncoder();
  private textDecoder = new TextDecoder("utf-8", { fatal: false });

  private constructor(
    config: TokenizerConfig,
    vocab: Map<string, number>,
    merges: [string, string][]
  ) {
    this.config = config;
    this.vocab = vocab;
    this.merges = merges;
    this.bosId = config.bosToken ?? 1;
    this.eosId = config.eosToken ?? 2;

    // Build reverse vocab
    this.reverseVocab = new Map();
    for (const [token, id] of vocab) {
      this.reverseVocab.set(id, token);
    }

    // Build merge rank lookup
    this.mergeRanks = new Map();
    for (let i = 0; i < merges.length; i++) {
      this.mergeRanks.set(`${merges[i][0]} ${merges[i][1]}`, i);
    }
  }

  /**
   * Create a tokenizer from GGUF metadata.
   * GGUF stores tokens and merge rules in the metadata.
   *
   * @param metadata - GGUF metadata containing `tokenizer.ggml.*` keys.
   * @returns A configured `Tokenizer` instance.
   *
   * @example
   * ```ts
   * const parser = new GGUFParser(buffer);
   * const gguf = parser.parse();
   * const tokenizer = Tokenizer.fromGGUFMetadata(gguf.metadata);
   * const ids = tokenizer.encode("Hello world");
   * ```
   */
  static fromGGUFMetadata(metadata: Record<string, unknown>): Tokenizer {
    const tokens = metadata["tokenizer.ggml.tokens"] as string[];
    const mergesRaw = metadata["tokenizer.ggml.merges"] as
      | string[]
      | undefined;
    const model = (metadata["tokenizer.ggml.model"] as string) ?? "gpt2";

    const vocab = new Map<string, number>();
    for (let i = 0; i < tokens.length; i++) {
      vocab.set(tokens[i], i);
    }

    const merges: [string, string][] = [];
    if (mergesRaw) {
      for (const m of mergesRaw) {
        const parts = m.split(" ");
        if (parts.length === 2) {
          merges.push([parts[0], parts[1]]);
        }
      }
    }

    const bosId =
      (metadata["tokenizer.ggml.bos_token_id"] as number) ?? 1;
    const eosId =
      (metadata["tokenizer.ggml.eos_token_id"] as number) ?? 2;

    const config: TokenizerConfig = {
      type: model === "gpt2" ? "bpe" : "sentencepiece",
      vocabSize: tokens.length,
      bosToken: bosId,
      eosToken: eosId,
    };

    return new Tokenizer(config, vocab, merges);
  }

  /**
   * Create a tokenizer from a vocab JSON object (e.g., `tokenizer.json`).
   *
   * @param data - Object with `vocab`, `merges`, and optional `config`.
   * @returns A configured `Tokenizer` instance.
   *
   * @example
   * ```ts
   * const resp = await fetch("https://example.com/tokenizer.json");
   * const tokenizer = Tokenizer.fromJSON(await resp.json());
   * ```
   */
  static fromJSON(data: {
    vocab: Record<string, number>;
    merges: string[];
    config?: Partial<TokenizerConfig>;
  }): Tokenizer {
    const vocab = new Map(Object.entries(data.vocab));
    const merges: [string, string][] = data.merges.map((m) => {
      const parts = m.split(" ");
      return [parts[0], parts[1]];
    });

    const config: TokenizerConfig = {
      type: data.config?.type ?? "bpe",
      vocabSize: vocab.size,
      bosToken: data.config?.bosToken ?? 1,
      eosToken: data.config?.eosToken ?? 2,
    };

    return new Tokenizer(config, vocab, merges);
  }

  /**
   * Encode text to token IDs.
   * @param text Input text
   * @param addBos Prepend BOS token
   */
  encode(text: string, addBos = true): Uint32Array {
    const tokens: number[] = [];

    if (addBos) {
      tokens.push(this.bosId);
    }

    if (this.config.type === "sentencepiece") {
      // SentencePiece: prepend space, use byte-level fallback
      text = " " + text;
    }

    // Pre-tokenize: split by regex (GPT-2 style)
    const pattern =
      /(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;
    const words = text.match(pattern) ?? [text];

    for (const word of words) {
      const wordTokens = this.bpeEncode(word);
      tokens.push(...wordTokens);
    }

    return new Uint32Array(tokens);
  }

  /**
   * Decode token IDs back to text.
   */
  decode(ids: Uint32Array | number[]): string {
    const parts: string[] = [];
    for (const id of ids) {
      if (id === this.bosId || id === this.eosId) continue;
      const token = this.reverseVocab.get(id);
      if (token !== undefined) {
        parts.push(this.decodeToken(token));
      }
    }
    return parts.join("");
  }

  /**
   * Decode a single token ID to its string representation.
   */
  decodeToken(token: string): string {
    // Handle byte-level tokens like <0x0A>
    if (token.startsWith("<0x") && token.endsWith(">")) {
      const byte = parseInt(token.slice(3, -1), 16);
      return String.fromCharCode(byte);
    }

    // SentencePiece uses ▁ (U+2581) for spaces
    if (this.config.type === "sentencepiece") {
      return token.replace(/▁/g, " ");
    }

    // GPT-2 style byte encoding
    return this.bytesToString(token);
  }

  get eosTokenId(): number {
    return this.eosId;
  }

  get bosTokenId(): number {
    return this.bosId;
  }

  /** Returns the vocab ID for `<|eot_id|>`, or undefined if not present. */
  get eotTokenId(): number | undefined {
    return this.vocab.get("<|eot_id|>");
  }

  /** Returns the vocab ID for `<|im_end|>`, or undefined if not present. */
  get imEndTokenId(): number | undefined {
    return this.vocab.get("<|im_end|>");
  }

  /**
   * Apply the appropriate chat template to a list of messages.
   * Auto-detects ChatML vs LLaMA 3 format from vocab.
   * Falls back to plain encode() if no special tokens are found.
   */
  applyChatTemplate(messages: ChatMessage[]): Uint32Array {
    // Check for ChatML tokens first
    const imStartId = this.vocab.get("<|im_start|>");
    const imEndId = this.vocab.get("<|im_end|>");
    if (imStartId !== undefined && imEndId !== undefined) {
      return this.applyChatML(messages, imStartId, imEndId);
    }

    // Check for LLaMA 3 tokens
    const startHeaderId = this.vocab.get("<|start_header_id|>");
    const endHeaderId = this.vocab.get("<|end_header_id|>");
    const eotId = this.vocab.get("<|eot_id|>");

    // Fall back to plain encoding if special tokens aren't in vocab
    if (startHeaderId === undefined || endHeaderId === undefined || eotId === undefined) {
      console.warn(`[0xBitNet] Chat template fallback: special tokens missing`);
      const text = messages.map((m) => m.content).join("\n");
      return this.encode(text);
    }

    console.debug(`[0xBitNet] Chat template: LLaMA 3 (start_header=${startHeaderId}, end_header=${endHeaderId}, eot=${eotId})`);

    const tokens: number[] = [this.bosId];

    for (const msg of messages) {
      tokens.push(startHeaderId);
      tokens.push(...this.encode(msg.role, false));
      tokens.push(endHeaderId);
      tokens.push(...this.encode("\n\n" + msg.content, false));
      tokens.push(eotId);
    }

    // Trailing assistant header to prompt generation
    tokens.push(startHeaderId);
    tokens.push(...this.encode("assistant", false));
    tokens.push(endHeaderId);
    tokens.push(...this.encode("\n\n", false));

    return new Uint32Array(tokens);
  }

  /**
   * Apply the ChatML template (used by Falcon-E and similar models).
   * Format: <|im_start|>role\ncontent<|im_end|>\n
   */
  private applyChatML(messages: ChatMessage[], imStartId: number, imEndId: number): Uint32Array {
    console.debug(`[0xBitNet] Chat template: ChatML (im_start=${imStartId}, im_end=${imEndId})`);

    const tokens: number[] = [this.bosId];
    for (const msg of messages) {
      tokens.push(imStartId);
      tokens.push(...this.encode(msg.role + "\n" + msg.content, false));
      tokens.push(imEndId);
      tokens.push(...this.encode("\n", false));
    }
    // Trailing assistant prompt
    tokens.push(imStartId);
    tokens.push(...this.encode("assistant\n", false));
    return new Uint32Array(tokens);
  }

  private bpeEncode(word: string): number[] {
    if (word.length === 0) return [];

    // Convert to initial symbols
    let symbols: string[];
    if (this.config.type === "sentencepiece") {
      symbols = [...word].map((c) => c.replace(" ", "▁"));
    } else {
      symbols = this.stringToBytes(word);
    }

    // Iteratively merge the most frequent pair
    while (symbols.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;

      for (let i = 0; i < symbols.length - 1; i++) {
        const pair = `${symbols[i]} ${symbols[i + 1]}`;
        const rank = this.mergeRanks.get(pair);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }

      if (bestIdx === -1) break;

      const merged = symbols[bestIdx] + symbols[bestIdx + 1];
      symbols.splice(bestIdx, 2, merged);
    }

    // Look up token IDs
    const ids: number[] = [];
    for (const sym of symbols) {
      const id = this.vocab.get(sym);
      if (id !== undefined) {
        ids.push(id);
      } else {
        // Byte-level fallback
        for (const byte of this.textEncoder.encode(sym)) {
          const byteToken = `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;
          const byteId = this.vocab.get(byteToken);
          if (byteId !== undefined) {
            ids.push(byteId);
          }
        }
      }
    }

    return ids;
  }

  // GPT-2 byte-to-unicode mapping
  private static byteToUnicode: Map<number, string> | null = null;

  private static getByteToUnicode(): Map<number, string> {
    if (Tokenizer.byteToUnicode) return Tokenizer.byteToUnicode;

    const map = new Map<number, string>();
    // Printable ASCII + Latin-1 supplement
    const ranges = [
      [33, 126],
      [161, 172],
      [174, 255],
    ];
    const bs: number[] = [];
    for (const [start, end] of ranges) {
      for (let i = start; i <= end; i++) {
        bs.push(i);
      }
    }
    const cs = [...bs];
    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) {
        bs.push(b);
        cs.push(256 + n);
        n++;
      }
    }
    for (let i = 0; i < bs.length; i++) {
      map.set(bs[i], String.fromCharCode(cs[i]));
    }

    Tokenizer.byteToUnicode = map;
    return map;
  }

  private stringToBytes(text: string): string[] {
    const b2u = Tokenizer.getByteToUnicode();
    const bytes = this.textEncoder.encode(text);
    const result: string[] = [];
    for (const byte of bytes) {
      result.push(b2u.get(byte) ?? String.fromCharCode(byte));
    }
    return result;
  }

  private bytesToString(token: string): string {
    const b2u = Tokenizer.getByteToUnicode();
    const u2b = new Map<string, number>();
    for (const [k, v] of b2u) {
      u2b.set(v, k);
    }

    const bytes: number[] = [];
    for (const ch of token) {
      const b = u2b.get(ch);
      if (b !== undefined) {
        bytes.push(b);
      } else {
        bytes.push(ch.charCodeAt(0));
      }
    }

    return this.textDecoder.decode(new Uint8Array(bytes));
  }
}
