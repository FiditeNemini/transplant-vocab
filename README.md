# Vocab Transplantation Tool

Transplants vocabulary between language models, enabling the creation of draft models for efficient speculative decoding **WITHOUT** retraining.

This tool allows you to combine the transformer architecture and weights from a donor model with the tokenizer of a target model, creating a hybrid model that can serve as a draft model in speculative decoding pipelines. By matching token-to-token or multi-token mappings between vocabularies, it intelligently transfers embeddings while preserving semantic relationships. This approach eliminates the need for expensive retraining or distillation procedures typically required for creating compatible draft models, making it an efficient solution for accelerating inference through speculative decoding techniques.

## Features

- Preserve the donor model's intelligence/performance.
- Adopt donor model to use the target model's tokenizer.

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- tqdm

## Usage

### Basic Command
```bash
python transplant_embeddings.py /path/to/donor_model /path/to/target_model /path/to/output_model
```

### Options

| Flag | Description |
|------|-------------|
| `--overwrite` | Replace existing output directory |
| `--unmapped-init-scale [0-1]` | Initialize unmapped output tokens with scaled mean embeddings (only useful if you plan to fine-tune) |
| `--use-cpu-only` | Use CPU instead of GPU with float32 precision |
| `--trust-remote-code` | Allow custom code execution when loading models with non-standard architectures |
| `--verbose` | Show detailed token mapping output |

### Example
```bash
# For direct use (no fine-tuning planned)
python transplant_embeddings.py ./Qwen2.5-0.5B-Instruct ./DeepSeek-R1 ./DeepSeek-R1-DRAFT-0.5B

# For creating a model you plan to fine-tune
python transplant_embeddings.py ./Qwen2.5-0.5B-Instruct ./DeepSeek-R1 ./DeepSeek-R1-DRAFT-0.5B --unmapped-init-scale 0.1
```

## Design Rationale

### Input Embeddings (Final Token Strategy)
When a target token maps to multiple donor tokens:
```text
Target: [X] → Donor: [A, B, C]
```
We use **C** (the final token) because:

1. Transformers process tokens sequentially, with transformer blocks "looking backward".
2. It's the transformer blocks that integrate context from previous tokens.
3. Taking the mean of all tokens doesn't align with how transformers process sequences.
4. Using the final token aligns with how the transformers process the previous token to create the next token.

### Output Head (First Token Uniqueness)
When a target token maps to multiple donor tokens:
```text
Target: [Y] → Donor: [D, E, F]
```
We use **D** (the first token) because:

1. The model decides on word endings in subsequent autoregressive passes.
2. Using mean embeddings would inappropriately include information about future word endings.
3. We track which first tokens have been used to avoid probability mass inflation.
4. When a first token is already used, we have two options:
   - Initialize to zero (default, best for direct use without fine-tuning).
   - Use a scaled mean of all token embeddings (with `--unmapped-init-scale`, only useful as a better starting point for fine-tuning).

### Mathematical Considerations

- Using means or scaling logits isn't mathematically ideal for probability distribution.
- Proper token splitting would require subtracting `log(n)` from each token in an n-token group.
- In the absence of an `lm_head.bias`, our approach provides the most practical solution.
- The `--unmapped-init-scale` option should only be used if you plan to fine-tune the model afterward, as it provides a better initialization point for training but may produce unreliable outputs if used directly.

## Technical Notes

- **CPU Option**: For systems without GPU or for models too large for your GPU (note: this will load/save as `float32`).
- **Multi-Token Mappings**: Statistics showing distribution of mapping types.
- **Output Head Initialization**: Shows percentage of tokens initialized with different strategies.
- **Fine-tuning Preparation**: Use `--unmapped-init-scale` when creating models for further training, leave at default 0.0 for direct use.

## Credit

Original concept by [turboderp](https://huggingface.co/turboderp). Based on [original implementation](https://huggingface.co/turboderp/Qwama-0.5B-Instruct/blob/main/vocab_transplant.py).

## License

Apache 2.0 License - See [LICENSE](LICENSE) for details
