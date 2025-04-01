#!/usr/bin/env python3
"""
Vocab Transplantation Tool

All credit to turboderp for the original idea:

https://huggingface.co/turboderp/Qwama-0.5B-Instruct/blob/main/vocab_transplant.py
"""

import argparse
import json
import os
import re
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
    parser.add_argument("--override", nargs = 2, action = "append", default = [],
                       help = "Override target token with donor sequence (can be used multiple times)")
    parser.add_argument("--weighting-decay-factor", type = float, default = 0.5,
                       help = "Decay factor [0-1] for multi-token mappings: "
                            "0=first token only, 0.5=decreasing weights, 1=uniform mean")
    parser.add_argument("--trim-layers", type = str,
                       help = "Trim out a range of layers from the model: start-end (inclusive)")
    parser.add_argument("--trim-hidden-size", type = int,
                       help = "Trim the hidden state dimension (and the number of heads as a result)")
    parser.add_argument("--trim-intermediate-size", type = int,
                       help = "Trim the intermediate dimension of the MLP blocks")
    parser.add_argument("--use-cpu-only", action = "store_true",
                       help = "Use CPU only for model loading and processing in float32")
    parser.add_argument("--trust-remote-code", action = "store_true",
                       help = "Allow custom code execution when loading models with non-standard architectures")
    parser.add_argument("--patch-missing-bos", action = "store_true",
                       help = "Patch `tokenizer_config.json` for models like `Qwen` which don't use any `<BOS>` token")
    parser.add_argument("--overwrite", action = "store_true",
                       help = "Overwrite output directory if it exists")
    parser.add_argument("--verbose", action = "store_true",
                       help = "Show detailed token mapping output")

    args = parser.parse_args()

    if not (0.0 <= args.weighting_decay_factor <= 1.0):
        sys.exit(f"Error: --weighting-decay-factor must be between 0.0 and 1.0 (got {args.weighting_decay_factor})")

    if args.trim_layers:
        try:
            start, end = map(int, args.trim_layers.split('-'))
            if start < 0 or end < start:
                sys.exit(f"Error: Invalid layer range: {args.trim_layers}. Format should be start-end with start ≥ 0 and end ≥ start")
        except ValueError:
            sys.exit(f"Error: Invalid layer range format: {args.trim_layers}. Format should be start-end (e.g., 3-8)")

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

def count_model_parameters(model) -> Tuple[int, int, int]:
    """
    Count the total number of parameters in a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Tuple of (total parameters, embedding and LM head parameters only,
                 parameters excluding embeddings and LM head)
    """
    total_params = 0
    embedding_params = 0
    non_embedding_params = 0

    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count

        # Separate embedding/LM head parameters from the rest
        if any(skip_name in name for skip_name in ['embed_tokens', 'lm_head']):
            embedding_params += param_count
        else:
            non_embedding_params += param_count

    return total_params, embedding_params, non_embedding_params

def has_config_value(config, key: str) -> bool:
    """Check if a key exists in a model configuration, checking both flat and nested structures"""
    if isinstance(config, dict):
        return key in config or ("text_config" in config and key in config["text_config"])
    return hasattr(config, key) or (hasattr(config, "text_config") and hasattr(config.text_config, key))

def get_config_value(config, key: str, default = ...):
    """Get a value from a model configuration, handling both flat and nested structures"""
    assert default is not ... or has_config_value(config, key), f"{key} not found in model configuration"
    if not has_config_value(config, key):
        return default
    if isinstance(config, dict):
        return config[key] if key in config else config["text_config"][key]
    return getattr(config, key) if hasattr(config, key) else getattr(config.text_config, key)

def set_config_value(config, key: str, value):
    """Set a value in a model configuration, updating both flat and nested structures if present"""
    assert has_config_value(config, key), f"{key} not found in model configuration"
    if isinstance(config, dict):
        if key in config:
            config[key] = value
        if "text_config" in config and key in config["text_config"]:
            config["text_config"][key] = value
    else:
        if hasattr(config, key):
            setattr(config, key, value)
        if hasattr(config, "text_config") and hasattr(config.text_config, key):
            setattr(config.text_config, key, value)

def process_automatic_token_overrides(target_tokenizer, donor_tokenizer, target_config, donor_config, existing_map = None):
    """
    Process automatic token overrides for special tokens.
    
    Args:
        target_tokenizer: The target tokenizer
        donor_tokenizer: The donor tokenizer
        target_config: The target model configuration
        donor_config: The donor model configuration
        
    Returns:
        Dictionary mapping target token IDs to donor token IDs
    """
    override_map = existing_map.copy() if existing_map else {}

    special_tokens = ['bos_token_id', 'eos_token_id', 'pad_token_id']
    print(f"\nProcessing {len(special_tokens)} automatic token overrides:")

    for token_attr in special_tokens:
        # First try to get from the tokenizer
        target_token_id = getattr(target_tokenizer, token_attr)
        donor_token_id = getattr(donor_tokenizer, token_attr)

        # Try to get from config if not found in tokenizer
        if target_token_id is None and has_config_value(target_config, token_attr):
            target_token_id = get_config_value(target_config, token_attr)
        if donor_token_id is None and has_config_value(donor_config, token_attr):
            donor_token_id = get_config_value(donor_config, token_attr)

        # Try to perform the automatic match
        if target_token_id is not None:
            if donor_token_id is not None:
                if target_token_id not in override_map:
                    target_token = target_tokenizer.convert_ids_to_tokens(target_token_id)
                    donor_token = donor_tokenizer.convert_ids_to_tokens(donor_token_id)
                    override_map[target_token_id] = torch.tensor([donor_token_id], dtype = torch.long)
                    print(f"✔ {repr(token_attr)} : {target_token_id} {repr(target_token)} → [{donor_token_id}] {repr(donor_token)}")
                else:
                    print(f"✘ {repr(token_attr)} : {target_token_id} is already mapped to [{override_map[target_token_id].item()}]")
            else:
                print(f"✘ {repr(token_attr)} : Not found for donor model")
        else:
            print(f"✘ {repr(token_attr)} : Not found for target model")

    return override_map

def process_manual_token_overrides(target_tokenizer, donor_tokenizer, manual_overrides, existing_map = None):
    """
    Process manual token overrides specified by the user.
    
    Args:
        target_tokenizer: The target tokenizer
        donor_tokenizer: The donor tokenizer
        manual_overrides: List of (target_token, donor_tokens) pairs
        existing_map: Existing override map to update (optional)
        
    Returns:
        Updated dictionary mapping target token IDs to donor token IDs
    """
    override_map = existing_map.copy() if existing_map else {}

    if not manual_overrides:
        return override_map

    print(f"\nProcessing {len(manual_overrides)} manual token overrides:")
    for target_token, donor_tokens in manual_overrides:
        # Encode target token and verify it's a single token
        target_id = target_tokenizer.encode(target_token, add_special_tokens = False)
        assert len(target_id) == 1, f"Target token '{target_token}' maps to {len(target_id)} tokens. Must be a 1 token."
        target_id = target_id[0]

        # Replace newline characters with the actual byte representation of a newline (0x0A)
        # NOTE: If you don't do this then it will get wrongly encoded as the "\\n" string literal
        if "\\n" in donor_tokens:
            donor_tokens = donor_tokens.replace("\\n", chr(10))

        # Get the IDs from the token string
        encoded = donor_tokenizer.encode(donor_tokens, add_special_tokens = False, return_tensors = "pt").flatten()
        assert encoded.numel() != 0, f"Donor token '{donor_tokens}' for target ID {target_id} encodes to 0 tokens."

        # Store the donor token IDs
        override_map[target_id] = encoded

        print(f"✔ {target_id:6d} : {repr(target_token)} → {encoded.tolist()} {repr(donor_tokens)}")

    return override_map

def compute_front_loaded_mean(v, weighting_decay_factor = 0.5):
    """
    Computes the "front-loaded" exponentially-weighted mean with a weighting decay factor.
    
    Parameters:
    - v: torch tensor with values
    - weighting_decay_factor: parameter in [0, 1] controlling how quickly weights decay for subsequent vectors
    
    Returns:
    - Weighted average tensor
    
    Special cases:
    - weighting_decay_factor=0   : Returns only the first vector (maximum front-loading)
    - weighting_decay_factor=0.5 : Applies weights 1, 0.5, 0.25, 0.125, ... (earlier vectors have more influence)
    - weighting_decay_factor=1   : Returns the uniform arithmetic mean (no front-loading)
    """
    # Assert that weighting_decay_factor is in the valid range [0, 1]
    assert 0 <= weighting_decay_factor <= 1, f"weighting_decay_factor must be in range [0, 1], got {weighting_decay_factor}"

    n = v.shape[0]

    if n == 1 or weighting_decay_factor == 0:
        return v[0]  # First (or only) vector only
    elif weighting_decay_factor == 1:
        return torch.mean(v, dim = 0)  # Arithmetic mean
    else:
        # Compute the weights using geometric progression
        decay_powers = torch.tensor([weighting_decay_factor ** i for i in range(n)], device = v.device)
        decay_powers = decay_powers.view(-1, *([1] * (v.dim() - 1)))
        weighted_sum = torch.sum(decay_powers * v, dim = 0)
        denominator = torch.sum(decay_powers)
        return weighted_sum / denominator

def transplant_tokens(model, donor_config, target_tokenizer, donor_tokenizer,
                      override_map, vocab_size, used_vocab_size,
                      weighting_decay_factor, verbose = False):
    """
    Transplant token embeddings from donor model to target vocabulary.
    
    Args:
        model: The donor model
        donor_config: The donor model configuration
        target_tokenizer: The target tokenizer
        donor_tokenizer: The donor tokenizer
        override_map: Dictionary mapping target token IDs to donor token IDs
        vocab_size: Total size of the target vocabulary
        used_vocab_size: Number of tokens actually used in the target vocabulary
        weighting_decay_factor: Factor for weighting multi-token mappings
        verbose: Whether to print detailed mapping information
        
    Returns:
        Tuple of (new state dict, embedding statistics)
    """
    # Get donor hidden size
    donor_hidden_size = get_config_value(donor_config, "hidden_size")

    # Get donor embeddings
    donor_embed_tokens = model.model.embed_tokens.weight
    if get_config_value(donor_config, "tie_word_embeddings", False):
        print("\nNOTE: Using an \"untied\" copy of 'embed_tokens.weight' as new 'lm_head.weight' tensor...\n")
        donor_lm_head = model.model.embed_tokens.weight
    else:
        print("\nNOTE: Using actual 'lm_head.weight' tensor as donor not configured with 'tie_word_embeddings...\n")
        donor_lm_head = model.lm_head.weight

    # Initialize new embedding and head tensors with zeros
    new_embed_tokens = torch.zeros(
        (vocab_size, donor_hidden_size),
        dtype = donor_embed_tokens.dtype,
        device = donor_embed_tokens.device
    )
    new_lm_head = torch.zeros(
        (vocab_size, donor_hidden_size),
        dtype = donor_lm_head.dtype,
        device = donor_lm_head.device
    )

    # Track mapping statistics
    mapping_counts = {}
    lm_head_copy_count = 0
    lm_head_mean_count = 0

    # Configure progress display
    iterator = range(used_vocab_size)
    if verbose:
        print("Transplanting tokens:")
    else:
        iterator = tqdm(iterator, desc = "Transplanting tokens", unit = "token")

    for idx in iterator:
        decoded = target_tokenizer.decode([idx], decode_special_tokens = True)
        if idx in override_map:
            encoded = override_map[idx]
        else:
            encoded = donor_tokenizer.encode(decoded, add_special_tokens = False, return_tensors = "pt").flatten()

        if verbose:
            print(f"- {idx:6d} : {repr(decoded)} → {encoded.tolist()}")

        # Track mapping types
        if encoded.numel() in mapping_counts:
            mapping_counts[encoded.numel()] += 1
        else:
            mapping_counts[encoded.numel()] = 1

        # Use only the final token of encoded sequence for input embeddings
        new_embed_tokens[idx] = donor_embed_tokens[encoded[-1]]

        # Use a "front-loaded" exponentially-weighted mean for lm_head embeddings
        if encoded.numel() == 1:
            new_lm_head[idx] = donor_lm_head[encoded[0].item()]
            lm_head_copy_count += 1
        else:
            head_embeddings = donor_lm_head[encoded.flatten()]
            new_lm_head[idx] = compute_front_loaded_mean(head_embeddings, weighting_decay_factor)
            lm_head_mean_count += 1

    # Print statistics
    print("\nTransplant mappings:")
    for count, occurrences in sorted(mapping_counts.items()):
        mapping_label = f"{count} to 1"
        print(f"- {mapping_label:<8}: {occurrences} ({(occurrences/used_vocab_size*100):.2g}%)")

    print("\nHead initialized with:")
    lm_head_zeroed_count = vocab_size - (lm_head_copy_count + lm_head_mean_count)
    print(f"- Copies : {lm_head_copy_count} ({(lm_head_copy_count/vocab_size*100):.2g}%)")
    print(f"- Means  : {lm_head_mean_count} ({(lm_head_mean_count/vocab_size*100):.2g}%)")
    print(f"- Zeros  : {lm_head_zeroed_count} ({(lm_head_zeroed_count/vocab_size*100):.2g}%)")

    # Make a copy of the model's state_dict and get the type
    new_state_dict = model.state_dict().copy()
    old_dtype = model.model.embed_tokens.weight.dtype

    # Update the state_dict with new embeddings
    new_state_dict['model.embed_tokens.weight'] = new_embed_tokens.to(dtype = old_dtype)
    new_state_dict['lm_head.weight'] = new_lm_head.to(dtype = old_dtype)

    return new_state_dict

def trim_model_layers(model, state_dict, start_layer, end_layer):
    """
    Trim out a range of layers from the model and its state_dict.
    
    Args:
        model: The model to modify
        state_dict: The state dictionary to modify
        start_layer: The first layer to remove (inclusive)
        end_layer: The last layer to remove (inclusive)
    
    Returns:
        Tuple of (modified model, modified state_dict)
    """
    # Get the total number of layers in the model
    total_layers = get_config_value(model.config, 'num_hidden_layers')
    assert start_layer >= 0 and start_layer < end_layer and end_layer < total_layers, f"Invalid layer range: start={start_layer}, end={end_layer}, total={total_layers}"

    print(f"\nTrimming layers {start_layer} through {end_layer} (inclusive): ")

    # Calculate how many layers to keep
    new_layer_count = total_layers - (end_layer - start_layer + 1)
    print(f"- Old layer count : {total_layers} (layers 0-{total_layers-1})")
    print(f"- New layer count : {new_layer_count} (keeping layers 0-{start_layer-1} and {end_layer+1}-{total_layers-1})")

    # Update the model configuration
    set_config_value(model.config, 'num_hidden_layers', new_layer_count)

    # Create a mapping from old layer indices to new layer indices
    layer_mapping = {}
    new_idx = 0
    for old_idx in range(total_layers):
        if old_idx < start_layer or old_idx > end_layer:
            layer_mapping[old_idx] = new_idx
            new_idx += 1

    # Create a new state dict with trimmed layers
    new_state_dict = {}
    removed_keys = []
    renamed_keys = []

    # First pass: identify all keys to process
    all_keys = list(state_dict.keys())

    layer_patterns = [r'model\.layers\.(\d+)\.', r'transformer\.h\.(\d+)\.', r'model\.decoder\.layers\.(\d+)\.']

    for key in all_keys:
        # Check if this key corresponds to a layer
        layer_match = None
        for pattern in layer_patterns:
            match = re.search(pattern, key)
            if match:
                layer_idx = int(match.group(1))
                if start_layer <= layer_idx <= end_layer:
                    # This layer should be removed
                    removed_keys.append(key)
                else:
                    # This layer is kept, but we need to renumber it
                    new_layer_idx = layer_mapping[layer_idx]
                    prefix = match.group(0)  # e.g., "model.layers.22."
                    new_prefix = prefix.replace(f"{layer_idx}", f"{new_layer_idx}")
                    new_key = key.replace(prefix, new_prefix)

                    # Add to renamed keys list
                    renamed_keys.append((key, new_key))

                    # Create a new tensor to avoid shared memory issues
                    new_state_dict[new_key] = state_dict[key].clone()

                # We found a match, so no need to check other patterns
                break

        # If no layer match was found, keep the tensor as is
        if layer_match is None and key not in removed_keys and not any(key == old_key for old_key, _ in renamed_keys):
            new_state_dict[key] = state_dict[key].clone()

    # For models with specific architectures, we might need to modify the layers list
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Create a new ModuleList with only the layers we want to keep
        new_layers = nn.ModuleList()
        for i, layer in enumerate(model.model.layers):
            if i < start_layer or i > end_layer:
                new_layers.append(layer)
        model.model.layers = new_layers

    print(f"- Removed {len(removed_keys)} tensors from state_dict")
    print(f"- Renamed {len(renamed_keys)} layer tensors to new indices")
    print(f"- Updated model configuration: num_hidden_layers = {new_layer_count}")

    return model, new_state_dict

def trim_tensors(state_dict, old_size, new_size):
    """
    Trim all tensors in a state dictionary that have dimensions matching old_size.
    
    Args:
        state_dict: The state dictionary to modify
        old_size: The original dimension size to look for
        new_size: The new dimension size to trim to
    
    Returns:
        Tuple of (new state dict, count of trimmed tensors)
    """
    new_state_dict = {}
    trimmed_count = 0

    # Process each tensor in the state dict
    for key, tensor in state_dict.items():
        # Check if this tensor has a dimension matching the size to trim
        if any(dim == old_size for dim in tensor.shape):
            # Create a new tensor with the appropriate dimensions
            new_shape = list(tensor.shape)
            for i, dim in enumerate(new_shape):
                if dim == old_size:
                    new_shape[i] = new_size

            # Create a completely new tensor with the new shape
            if len(new_shape) == 1:
                new_tensor = torch.zeros(
                    new_shape[0],
                    dtype = tensor.dtype,
                    device = tensor.device
                )
                # Copy data from the original tensor
                new_tensor[:] = tensor[:new_size]
            elif len(new_shape) == 2:
                new_tensor = torch.zeros(
                    new_shape[0], new_shape[1],
                    dtype = tensor.dtype,
                    device = tensor.device
                )
                # Copy data based on which dimensions need trimming
                if tensor.shape[0] == old_size and tensor.shape[1] == old_size:
                    new_tensor[:,:] = tensor[:new_size,:new_size]
                elif tensor.shape[0] == old_size:
                    new_tensor[:,:] = tensor[:new_size,:]
                else:
                    new_tensor[:,:] = tensor[:,:new_size]
            else:
                # For higher dimensional tensors
                new_tensor = torch.zeros(
                    new_shape,
                    dtype = tensor.dtype,
                    device = tensor.device
                )
                # Create slices for copying
                src_slices = tuple(slice(0, new_shape[i]) if tensor.shape[i] == old_size else slice(None)
                                  for i in range(len(tensor.shape)))
                dst_slices = tuple(slice(None) for _ in range(len(tensor.shape)))
                new_tensor[dst_slices] = tensor[src_slices]

            new_state_dict[key] = new_tensor
            trimmed_count += 1
        else:
            # Keep tensors that don't have matching dimensions unchanged
            new_state_dict[key] = tensor.clone()

    return new_state_dict, trimmed_count

def trim_model_hidden_size(model, state_dict, new_size):
    """
    Trim the hidden state dimension of the residual stream.
    
    Args:
        model: The model to modify
        state_dict: The state dictionary to modify
        new_size: The new hidden state dimension to use
    
    Returns:
        Tuple of (modified model, modified state_dict)
    """
    old_size = get_config_value(model.config, 'hidden_size')
    old_num_heads = get_config_value(model.config, 'num_attention_heads')
    old_num_kv_heads = get_config_value(model.config, 'num_key_value_heads')
    assert new_size < old_size, f"New hidden size ({new_size}) must be smaller than old ({old_size})"
    assert old_size % old_num_heads == 0, f"Old hidden size ({old_size}) is not divisible by number of heads ({old_num_heads})"

    head_dimension = old_size // old_num_heads
    assert new_size % head_dimension == 0, f"New hidden size ({new_size}) is not divisible by head dimension ({head_dimension})"

    new_num_heads = new_size // head_dimension
    assert new_num_heads >= old_num_kv_heads, f"New num heads({new_num_heads}) is less than KV heads {old_num_kv_heads})"

    print(f"\nTrimming hidden size from {old_size} to {new_size}: ")
    print(f"- Old hidden size : {old_size}")
    print(f"- New hidden size : {new_size}")

    set_config_value(model.config, 'hidden_size', new_size)
    set_config_value(model.config, 'num_attention_heads', new_num_heads)
    print(f"- Updated model configuration: hidden_size = {new_size}")
    print(f"- Updated model configuration: num_attention_heads = {new_num_heads}")

    new_state_dict, trimmed_count = trim_tensors(state_dict, old_size, new_size)
    print(f"- Trimmed {trimmed_count} tensors in state_dict")

    return model, new_state_dict

def trim_model_intermediate_size(model, state_dict, new_size):
    """
    Trim the hidden state dimension of the MLP blocks.
    
    Args:
        model: The model to modify
        state_dict: The state dictionary to modify
        new_size: The new hidden state dimension to use
    
    Returns:
        Tuple of (modified model, modified state_dict)
    """
    old_size = get_config_value(model.config, 'intermediate_size')
    assert new_size < old_size, f"New intermediate size ({new_size}) must be smaller than old ({old_size})"

    print(f"\nTrimming intermediate size from {old_size} to {new_size}: ")
    print(f"- Old intermediate size : {old_size}")
    print(f"- New intermediate size : {new_size}")

    set_config_value(model.config, 'intermediate_size', new_size)
    print(f"- Updated model configuration: intermediate_size = {new_size}")

    new_state_dict, trimmed_count = trim_tensors(state_dict, old_size, new_size)
    print(f"- Trimmed {trimmed_count} tensors in state_dict")

    return model, new_state_dict

def patch_tokenizer_config_bos(output_dir):
    """
    Patch the tokenizer configuration to handle models without BOS tokens.
    
    Args:
        output_dir: Path to the output directory containing the tokenizer_config.json
    """
    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        print(f"\nPatching BOS handling in '{tokenizer_config_path}'")
        try:
            # Read the file as text without specifying encoding
            with open(tokenizer_config_path, "r") as f:
                config_text = f.read()

            # Make sure that add_bos_token is set to false
            config_text = config_text.replace('"add_bos_token": true', '"add_bos_token": false')
            print("- Updated 'add_bos_token' configuration.")

            # Remove any use of bos_token from chat template
            # NOTE: We can't (safely) set '"bos_token": null', but it shouldn't matter with these two patches...
            config_text = config_text.replace("{{ bos_token }}", "").replace("{{bos_token}}", "")
            print("- Removed all references to 'bos_token' from Jinja chat template.")

            # Write the modified text back without specifying encoding
            with open(tokenizer_config_path, "w") as f:
                f.write(config_text)
        except Exception as e:
            print(f"Warning: Failed to patch tokenizer configuration: {e}")

def debug_model_tensors(model, state_dict):
    """
    Print detailed information about model parameters and state dict tensors
    to help debug shape mismatches.
    
    Args:
        model: The model to inspect
        state_dict: The state dictionary to inspect
    """
    print("\n=== MODEL PARAMETERS ===")
    for name, param in model.named_parameters():
        print(f"{name}: shape={param.shape}, dtype={param.dtype}")

    print("\n=== STATE DICT TENSORS ===")
    for key, tensor in state_dict.items():
        print(f"{key}: shape={tensor.shape}, dtype={tensor.dtype}")

    print("\n=== SHAPE MISMATCHES ===")
    mismatches = []
    for name, param in model.named_parameters():
        if name in state_dict and param.shape != state_dict[name].shape:
            mismatches.append((name, param.shape, state_dict[name].shape))

    if mismatches:
        print("Found shape mismatches between model parameters and state dict:")
        for name, model_shape, dict_shape in mismatches:
            print(f"- {name}: model={model_shape}, state_dict={dict_shape}")
    else:
        print("No shape mismatches found between model parameters and state dict.")

    # Check for tensors in state_dict that don't exist in model
    model_params = {name for name, _ in model.named_parameters()}
    extra_tensors = {key for key in state_dict if key not in model_params}
    if extra_tensors:
        print("\n=== EXTRA TENSORS IN STATE DICT ===")
        for key in extra_tensors:
            print(f"{key}: shape={state_dict[key].shape}")

    # Check for parameters in model that don't exist in state_dict
    missing_tensors = {name for name, _ in model.named_parameters() if name not in state_dict}
    if missing_tensors:
        print("\n=== MISSING TENSORS IN STATE DICT ===")
        for name in missing_tensors:
            print(name)

def main():
    args = parse_arguments()
    validate_directories(args)

    # Load configurations
    donor_config = load_model_config(args.donor_dir)
    target_config = load_model_config(args.target_dir)

    # Get configuration values we will need
    target_vocab_size = get_config_value(target_config, "vocab_size")
    donor_vocab_size = get_config_value(donor_config, "vocab_size")
    donor_num_layers = get_config_value(donor_config, 'num_hidden_layers')
    donor_tied_embeddings = get_config_value(donor_config, "tie_word_embeddings", False)
    donor_hidden_size = get_config_value(donor_config, "hidden_size")
    donor_num_heads = get_config_value(donor_config, "num_attention_heads")
    donor_intermediate_size = get_config_value(donor_config, "intermediate_size")

    # Load tokenizers
    donor_tokenizer = load_tokenizer(args.donor_dir, args.trust_remote_code)
    target_tokenizer = load_tokenizer(args.target_dir, args.trust_remote_code)

    # Load the donor model
    if args.use_cpu_only:
        model = load_model(args.donor_dir, args.trust_remote_code)
    else:
        model = load_model(args.donor_dir, args.trust_remote_code, get_config_value(donor_config, "torch_dtype", None))

    # The config file counts the all tokens, but we also need to know how many are used for the loop
    used_target_vocab_size = max(target_tokenizer.vocab.values()) + 1
    unused_target_vocab_size = target_vocab_size - used_target_vocab_size

    # Count parameters in donor model
    donor_total_params, donor_embedding_params, donor_non_embedding_params = count_model_parameters(model)
    donor_total_params_b = donor_total_params / 1e9
    donor_embedding_params_b = donor_embedding_params / 1e9
    donor_non_embedding_params_b = donor_non_embedding_params / 1e9

    print("\nInput model configuration:")
    print(f"- Target vocabulary size    : {target_vocab_size} (used = {used_target_vocab_size}, unused = {unused_target_vocab_size})")
    print(f"- Donor vocabulary size     : {donor_vocab_size}")
    print(f"- Donor num layers          : {donor_num_layers} (tied embeddings = {donor_tied_embeddings})")
    print(f"- Donor hidden size         : {donor_hidden_size}")
    print(f"- Donor attention heads     : {donor_num_heads}")
    print(f"- Donor intermediate size   : {donor_intermediate_size} (ratio = 1:{donor_intermediate_size/donor_hidden_size:.1f})")
    print(f"- Donor total parameters    : {donor_total_params} ({donor_total_params_b:.2f}B)")
    print(f"-- Embedding parameters     : {donor_embedding_params} ({donor_embedding_params_b:.2f}B)")
    print(f"-- Non-embedding parameters : {donor_non_embedding_params} ({donor_non_embedding_params_b:.2f}B)")

    # Automatic and manual overrides
    override_map = {}

    # Process automatic and manual token overrides
    override_map = process_automatic_token_overrides(target_tokenizer, donor_tokenizer, target_config, donor_config)

    # Process manual token overrides
    override_map = process_manual_token_overrides(target_tokenizer, donor_tokenizer, args.override, override_map)

    # Transplant tokens from donor model to target vocabulary
    new_state_dict = transplant_tokens(
        model = model,
        donor_config = donor_config,
        target_tokenizer = target_tokenizer,
        donor_tokenizer = donor_tokenizer,
        override_map = override_map,
        vocab_size = target_vocab_size,
        used_vocab_size = used_target_vocab_size,
        weighting_decay_factor = args.weighting_decay_factor,
        verbose = args.verbose
    )

    # Trim layers if requested
    if args.trim_layers:
        start_layer, end_layer = map(int, args.trim_layers.split('-'))
        model, new_state_dict = trim_model_layers(model, new_state_dict, start_layer, end_layer)

    # Trim hidden size if requested
    if args.trim_hidden_size:
        model, new_state_dict = trim_model_hidden_size(model, new_state_dict, args.trim_hidden_size)

    # Trim intermediate size if requested
    if args.trim_intermediate_size:
        model, new_state_dict = trim_model_intermediate_size(model, new_state_dict, args.trim_intermediate_size)

    # Update model architecture
    model.model.embed_tokens.num_embeddings = target_vocab_size
    model.lm_head.out_features = target_vocab_size

    # Update model config
    set_config_value(model.config, 'vocab_size', target_vocab_size)
    set_config_value(model.config, 'bos_token_id', target_tokenizer.bos_token_id)
    set_config_value(model.config, 'eos_token_id', target_tokenizer.eos_token_id)

    # Update the config's pad_token_id if it exists
    if has_config_value(model.config, 'pad_token_id'):
        if target_tokenizer.pad_token_id is not None:
            set_config_value(model.config, 'pad_token_id', target_tokenizer.pad_token_id)
        else:
            set_config_value(model.config, 'pad_token_id', target_tokenizer.eos_token_id)  # Default to EOS if no PAD to copy

    # Set the config's tie_word_embeddings to False if it exists
    if has_config_value(model.config, 'tie_word_embeddings'):
        set_config_value(model.config, 'tie_word_embeddings', False)

    # Re-initialize the model with the updated configuration and load into it the new state dict
    # NOTE: This seems to be more robust that just altering the model and state dict parameters
    model = type(model)(model.config)
    model.load_state_dict(new_state_dict)

    output_num_layers = get_config_value(model.config, 'num_hidden_layers')
    output_tied_embeddings = get_config_value(model.config, "tie_word_embeddings", False)
    output_hidden_size = get_config_value(model.config, "hidden_size")
    output_num_heads = get_config_value(model.config, "num_attention_heads")
    output_intermediate_size = get_config_value(model.config, "intermediate_size")

    # Count parameters in output model
    output_total_params, output_embedding_params, output_non_embedding_params = count_model_parameters(model)
    output_total_params_b = output_total_params / 1e9
    output_embedding_params_b = output_embedding_params / 1e9
    output_non_embedding_params_b = output_non_embedding_params / 1e9

    # Print output model configuration values
    print("\nOutput model configuration:")
    print(f"- Output vocabulary size    : {target_vocab_size}")
    print(f"- Output num layers         : {output_num_layers} (tied embeddings = {output_tied_embeddings})")
    print(f"- Output hidden size        : {output_hidden_size}")
    print(f"- Output attention heads    : {output_num_heads}")
    print(f"- Output intermediate size  : {output_intermediate_size} (ratio = 1:{output_intermediate_size/output_hidden_size:.1f})")
    print(f"- Output total parameters   : {output_total_params} ({output_total_params_b:.2f}B)")
    print(f"-- Embedding parameters     : {output_embedding_params} ({output_embedding_params_b:.2f}B)")
    print(f"-- Non-embedding parameters : {output_non_embedding_params} ({output_non_embedding_params_b:.2f}B)")

    # debug_model_tensors(model, new_state_dict)

    # Save final model and tokenizer
    print(f"\nSaving model and tokenizer to '{args.output_dir}' folder")
    model.save_pretrained(args.output_dir, state_dict = new_state_dict, safe_serialization = True)
    target_tokenizer.save_pretrained(args.output_dir)

    # Finally, attempt to patch the EOS stuff if the donor tokenizer doesn't use BOS tokens
    if args.patch_missing_bos and (getattr(donor_tokenizer, "add_bos_token", False)
                                   or getattr(donor_tokenizer, "bos_token", None) is None):
        patch_tokenizer_config_bos(args.output_dir)

    # TODO: Figure out why it causes a segmentation fault on exit???
    print("\nOperation completed successfully (ignore any 'segmentation fault' that follows!!!)")

if __name__ == "__main__":
    main()
