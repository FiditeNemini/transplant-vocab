#!/usr/bin/env python3
"""
Vocab Transplantation Tool

All credit to turboderp for the original idea:

https://huggingface.co/turboderp/Qwama-0.5B-Instruct/blob/main/vocab_transplant.py
"""

import argparse
import json
import os
import shutil
import sys
from typing import Tuple, Dict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import torch.nn as nn

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description = "Transplant token embeddings between language models",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("donor_dir", help = "Path to donor model directory")
    parser.add_argument("target_dir", help = "Path to target model directory")
    parser.add_argument("output_dir", help = "Path to output model directory")
    parser.add_argument("--overwrite", action = "store_true",
                       help = "Overwrite output directory if it exists")
    parser.add_argument("--unmapped-init-scale", type = float, default = 0.0,
                       help = "Scale factor [0-1] for initializing unmapped lm_head tokens")
    parser.add_argument("--use-cpu-only", action = "store_true",
                       help = "Use CPU only for model loading and processing in float32")
    parser.add_argument("--trust-remote-code", action = "store_true",
                       help = "Allow custom code execution when loading models with non-standard architectures")
    parser.add_argument("--verbose", action = "store_true",
                       help = "Show detailed token mapping output")

    args = parser.parse_args()

    if not (0.0 <= args.unmapped_init_scale <= 1.0):
        sys.exit(f"Error: --unmapped-init-scale must be between 0.0 and 1.0 (got {args.unmapped_init_scale})")

    return args

def validate_directories(args: argparse.Namespace) -> None:
    """Validate input/output directory structure and permissions"""
    for dir_type, dir_path in [("donor", args.donor_dir), ("target", args.target_dir)]:
        if not os.path.isdir(dir_path):
            sys.exit(f"Error: {dir_type} directory does not exist: {dir_path}")
        if not os.access(dir_path, os.R_OK):
            sys.exit(f"Error: No read permissions for {dir_type} directory: {dir_path}")

    if os.path.exists(args.output_dir):
        if args.overwrite:
            if not os.access(args.output_dir, os.W_OK):
                sys.exit(f"Error: No write permissions for output directory: {args.output_dir}")
            shutil.rmtree(args.output_dir)
        else:
            sys.exit(f"Error: Output directory exists (use --overwrite to replace): {args.output_dir}")

    try:
        os.makedirs(args.output_dir, exist_ok = True)
    except OSError as e:
        sys.exit(f"Error: Failed to create output directory: {e}")

def load_model_config(path: str) -> dict:
    """Load model configuration"""
    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        sys.exit(f"Error: Config file not found at {config_path}")

    try:
        print(f"Loading config from '{path}'... ", end = "")
        with open(config_path, "r", encoding = "utf-8") as f:
            config = json.load(f)
        print("Done.")
    except Exception as e:
        sys.exit(f"Error loading config from {config_path}: {e}")

    return config

def load_tokenizer(path: str, trust_remote_code = False) -> AutoTokenizer:
    """Load tokenizer with error handling"""
    try:
        print(f"Loading tokenizer from '{path}'... ", end = "")
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code = trust_remote_code)
        print("Done.")
        return tokenizer
    except Exception as e:
        sys.exit(f"Failed to load tokenizer: {e}")

def load_model(path: str, trust_remote_code = False, torch_dtype = None) -> AutoModelForCausalLM:
    """Load model with error handling"""
    try:
        print(f"Loading model from '{path}'... ", end = "")
        if torch_dtype is not None:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map = "auto",
                trust_remote_code = trust_remote_code,
                torch_dtype = torch_dtype
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code = trust_remote_code,
                device_map = "cpu"  # Will also load (and save) as torch.float32
            )
        print("Done.")
        return model
    except Exception as e:
        sys.exit(f"Failed to load model: {e}")

def main():
    args = parse_arguments()
    validate_directories(args)

    # Load configurations
    donor_config = load_model_config(args.donor_dir)
    target_config = load_model_config(args.target_dir)

    # Retrieving the donor vocabulary size (for both flat and nested configurations)
    if "text_config" in donor_config and "vocab_size" in donor_config["text_config"]:
        donor_vocab_size = donor_config["text_config"]["vocab_size"]
    else:
        assert "vocab_size" in donor_config, "vocab_size not found in source model config"
        donor_vocab_size = donor_config["vocab_size"]
    print(f"Donor vocab size: {donor_vocab_size}")

    # Retrieving the target vocabulary size (for both flat and nested configurations)
    if "text_config" in target_config and "vocab_size" in target_config["text_config"]:
        target_vocab_size = target_config["text_config"]["vocab_size"]
    else:
        assert "vocab_size" in target_config, "vocab_size not found in target model config"
        target_vocab_size = target_config["vocab_size"]
    print(f"Target vocab size: {target_vocab_size}")

    assert "hidden_size" in donor_config, "hidden_size not found in donor model config"

    # Load tokenizers
    donor_tokenizer = load_tokenizer(args.donor_dir, args.trust_remote_code)
    target_tokenizer = load_tokenizer(args.target_dir, args.trust_remote_code)

    # Load the donor model
    if args.use_cpu_only:
        model = load_model(args.donor_dir, args.trust_remote_code)
    else:
        model = load_model(args.donor_dir, args.trust_remote_code, donor_config.get("torch_dtype", None))

    # NOTE: The config file is often wrong, so get calculate from the tokenizer instead
    actual_target_vocab_size = max(target_tokenizer.vocab.values()) + 1

    # Initialize new embeddings
    donor_embed_tokens = model.model.embed_tokens.weight
    donor_lm_head = model.model.embed_tokens.weight if donor_config.get("tie_word_embeddings", False) else model.lm_head.weight

    new_embed_tokens = torch.zeros(
        (target_vocab_size, donor_config["hidden_size"]),
        dtype = donor_embed_tokens.dtype,
        device = donor_embed_tokens.device
    )
    new_lm_head = torch.zeros(
        (target_vocab_size, donor_config["hidden_size"]),
        dtype = donor_lm_head.dtype,
        device = donor_lm_head.device
    )

    # Track mapping statistics
    mapping_counts = {}

    # Track lm_head statistics
    lm_head_set_count = 0
    lm_head_scaled_count = 0

    # Used to track already used prefix tokens for the lm_head
    used_prefix_tokens = set()

    # Configure progress display
    iterator = range(actual_target_vocab_size)
    if not args.verbose:
        iterator = tqdm(iterator, desc = "Transplanting tokens", unit = "token")

    for idx in iterator:
        decoded = target_tokenizer.decode([idx], decode_special_tokens = True)
        encoded = donor_tokenizer.encode(decoded, add_special_tokens = False, return_tensors = "pt").flatten()

        if args.verbose:
            print(f"{idx:5d}: {repr(decoded)} → {encoded.tolist()}")

        # Track mapping types
        if encoded.numel() in mapping_counts:
            mapping_counts[encoded.numel()] += 1
        else:
            mapping_counts[encoded.numel()] = 1

        # Use only the final token of encoded sequence for input embeddings
        new_embed_tokens[idx] = donor_embed_tokens[encoded[-1]]

        # Use only the first token for head embeddings (unless asked to use scaled mean)
        prefix_token = encoded[0].item()
        if prefix_token not in used_prefix_tokens:
            used_prefix_tokens.add(prefix_token)
            new_lm_head[idx] = donor_lm_head[prefix_token]
            lm_head_set_count += 1
        elif args.unmapped_init_scale > 0:
            encode_tokens = encoded.flatten()
            head_embeddings = donor_lm_head[encode_tokens]
            new_lm_head[idx] = head_embeddings.mean(dim = 0) * args.unmapped_init_scale
            lm_head_scaled_count += 1

    # Print statistics
    print("\nTransplant mappings:")
    for count, occurrences in sorted(mapping_counts.items()):
        mapping_label = f"{count} to 1"
        print(f"- {mapping_label:<8}: {occurrences} ({occurrences/actual_target_vocab_size:.1%})")

    print("\nHead initialized with:")
    lm_head_zeroed_count = target_vocab_size - (lm_head_set_count + lm_head_scaled_count)
    print(f"- Copies : {lm_head_set_count} ({lm_head_set_count/target_vocab_size:.1%})")
    if lm_head_scaled_count > 0:
        print(f"- Means  : {lm_head_scaled_count} ({lm_head_scaled_count/target_vocab_size:.1%})")
    print(f"- Zeros  : {lm_head_zeroed_count} ({lm_head_zeroed_count/target_vocab_size:.1%})")

    # Make a copy of the model's state_dict and get the type
    new_state_dict = model.state_dict().copy()
    old_dtype = model.model.embed_tokens.weight.dtype

    # Update the state_dict with new embeddings
    new_state_dict['model.embed_tokens.weight'] = new_embed_tokens.to(dtype = old_dtype)
    new_state_dict['lm_head.weight'] = new_lm_head.to(dtype = old_dtype)

    # Update model architecture
    model.model.embed_tokens.num_embeddings = target_vocab_size
    model.lm_head.out_features = target_vocab_size

    # Update model config
    model.config.update({
        'vocab_size': target_vocab_size,
        'bos_token_id': target_tokenizer.bos_token_id,
        'eos_token_id': target_tokenizer.eos_token_id,
    })

    # Add pad_token_id if it exists in the target tokenizer
    if target_tokenizer.pad_token_id is not None:
        model.config.update({'pad_token_id': target_tokenizer.pad_token_id})

    # Set tie_word_embeddings to False if it exists
    if hasattr(model.config, 'tie_word_embeddings'):
        model.config.update({'tie_word_embeddings': False})

    # Save final model and tokenizer
    print(f"\nSaving model and tokenizer to {args.output_dir}")
    model.save_pretrained(args.output_dir, state_dict = new_state_dict, safe_serialization = True)
    target_tokenizer.save_pretrained(args.output_dir)

    print("Operation completed successfully")

if __name__ == "__main__":
    main()
