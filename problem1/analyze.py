"""
Analysis and visualization of attention patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from tqdm import tqdm

from model import Seq2SeqTransformer
from dataset import create_dataloaders, get_vocab_size
from attention import create_causal_mask


def extract_attention_weights(model, dataloader, device, num_samples=100):
    """
    Extract attention weights from model for analysis.

    Args:
        model: Trained transformer model
        dataloader: Data loader
        device: Device to run on
        num_samples: Number of samples to analyze

    Returns:
        Dictionary containing attention weights and sample data
    """
    model.eval()

    # Store attention weights per sample: [sample_idx][layer_idx] -> attention_weights
    all_encoder_attentions = []  # Will be list of lists: [sample][layer]
    all_decoder_self_attentions = []
    all_decoder_cross_attentions = []
    all_inputs = []
    all_targets = []

    samples_collected = 0

    with torch.no_grad():
        for batch in dataloader:
            if samples_collected >= num_samples:
                break

            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            batch_size = inputs.size(0)

            # For now, we'll need to hook into the attention layers
            encoder_attentions_by_layer = []  # Each element is [batch_size, num_heads, seq_len, seq_len]
            decoder_self_attentions_by_layer = []
            decoder_cross_attentions_by_layer = []

            # Register hooks to capture attention weights
            def make_hook(attention_list):
                def hook(module, input, output):
                    # output is (attention_output, attention_weights)
                    # attention_weights shape: [batch_size, num_heads, seq_len, seq_len]
                    attention_list.append(output[1].detach().cpu())
                return hook

            # Register hooks on attention layers
            encoder_hooks = []
            decoder_self_hooks = []
            decoder_cross_hooks = []
            
            # Hook encoder layers
            for layer in model.encoder_layers:
                hook = make_hook(encoder_attentions_by_layer)
                encoder_hooks.append(layer.self_attn.register_forward_hook(hook))
            
            # Hook decoder layers
            for layer in model.decoder_layers:
                self_hook = make_hook(decoder_self_attentions_by_layer)
                cross_hook = make_hook(decoder_cross_attentions_by_layer)
                decoder_self_hooks.append(layer.self_attn.register_forward_hook(self_hook))
                decoder_cross_hooks.append(layer.cross_attn.register_forward_hook(cross_hook))

            # Forward pass
            # Prepare decoder input for teacher forcing
            decoder_input = targets[:, :-1]
            tgt_mask = create_causal_mask(decoder_input.size(1), device=device)
            
            # Run model forward pass
            outputs = model(inputs, decoder_input, tgt_mask=tgt_mask)

            # Collect samples
            samples_to_take = min(batch_size, num_samples - samples_collected)
            all_inputs.extend(inputs[:samples_to_take].cpu().numpy())
            all_targets.extend(targets[:samples_to_take].cpu().numpy())

            # Extract attention weights per sample
            # encoder_attentions_by_layer is list of tensors: [layer_0_weights, layer_1_weights, ...]
            # Each tensor is [batch_size, num_heads, seq_len, seq_len]
            for sample_in_batch_idx in range(samples_to_take):
                # Initialize attention lists for this sample
                sample_encoder_attns = []
                sample_decoder_self_attns = []
                sample_decoder_cross_attns = []
                
                # Extract this sample's attention from each layer
                for layer_attn in encoder_attentions_by_layer:
                    # layer_attn shape: [batch_size, num_heads, seq_len, seq_len]
                    sample_attn = layer_attn[sample_in_batch_idx]  # [num_heads, seq_len, seq_len]
                    sample_encoder_attns.append(sample_attn)
                
                for layer_attn in decoder_self_attentions_by_layer:
                    sample_attn = layer_attn[sample_in_batch_idx]  # [num_heads, seq_len, seq_len]
                    sample_decoder_self_attns.append(sample_attn)
                
                for layer_attn in decoder_cross_attentions_by_layer:
                    sample_attn = layer_attn[sample_in_batch_idx]  # [num_heads, seq_len, seq_len]
                    sample_decoder_cross_attns.append(sample_attn)
                
                all_encoder_attentions.append(sample_encoder_attns)
                all_decoder_self_attentions.append(sample_decoder_self_attns)
                all_decoder_cross_attentions.append(sample_decoder_cross_attns)

            # Remove hooks
            for hook in encoder_hooks + decoder_self_hooks + decoder_cross_hooks:
                hook.remove()

            samples_collected += samples_to_take

    return {
        'encoder_attention': all_encoder_attentions,  # List of lists: [sample][layer]
        'decoder_self_attention': all_decoder_self_attentions,
        'decoder_cross_attention': all_decoder_cross_attentions,
        'inputs': all_inputs,
        'targets': all_targets
    }


def visualize_attention_pattern(attention_weights, input_tokens, output_tokens,
                               title="Attention Pattern", save_path=None):
    """
    Visualize attention weights as heatmap.

    Args:
        attention_weights: Attention weights [num_heads, out_len, in_len]
        input_tokens: Input token labels
        output_tokens: Output token labels
        title: Plot title
        save_path: Path to save figure
    """
    num_heads = attention_weights.shape[0]

    # Create figure with subplots for each head
    fig, axes = plt.subplots(
        2, (num_heads + 1) // 2,
        figsize=(5 * ((num_heads + 1) // 2), 8)
    )
    axes = axes.flatten()

    for head_idx in range(num_heads):
        ax = axes[head_idx]

        # Extract single head attention matrix
        head_attn = attention_weights[head_idx]
        
        # Ensure it's numpy and 2D
        if isinstance(head_attn, torch.Tensor):
            head_attn = head_attn.detach().cpu().numpy()
        
        # Remove any extra dimensions
        while len(head_attn.shape) > 2:
            head_attn = head_attn.squeeze()
        
        # Ensure it's exactly 2D
        if len(head_attn.shape) != 2:
            raise ValueError(f"Expected 2D attention matrix for heatmap, got shape {head_attn.shape}")

        # Plot heatmap
        sns.heatmap(
            head_attn,
            ax=ax,
            cmap='Blues',
            cbar=True,
            square=True,
            xticklabels=input_tokens,
            yticklabels=output_tokens,
            vmin=0,
            vmax=1
        )

        ax.set_title(f'Head {head_idx + 1}')
        ax.set_xlabel('Input Position')
        ax.set_ylabel('Output Position')

    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory and avoid blocking


def analyze_head_specialization(attention_data, output_dir):
    """
    Analyze what each attention head specializes in.

    Args:
        attention_data: Dictionary with attention weights and samples
        output_dir: Directory to save analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze encoder self-attention
    print("Analyzing encoder self-attention patterns...")

    # TODO: For each head, compute statistics:
    # - Average attention to operator token
    # - Average attention to same position (diagonal)
    # - Average attention to carry positions
    # - Entropy of attention distribution

    head_stats = {}

    # Get encoder attention weights
    encoder_attentions = attention_data['encoder_attention']
    if not encoder_attentions:
        print("No encoder attention weights found")
        return head_stats
    
    # Analyze each head
    num_heads = encoder_attentions[0][0].shape[0]  # [num_heads, seq_len, seq_len]
    
    for head_idx in range(num_heads):
        head_attentions = []
        
        # Collect attention weights for this head across all samples
        for sample_idx in range(len(encoder_attentions)):
            for layer_idx in range(len(encoder_attentions[sample_idx])):
                head_attentions.append(encoder_attentions[sample_idx][layer_idx][head_idx])  # [seq_len, seq_len]
        
        # Stack all attention weights for this head
        head_attentions = torch.stack(head_attentions)  # [num_samples * num_layers, seq_len, seq_len]
        
        # Compute statistics
        stats = {}
        
        # Average attention to operator token (position 3 in 3-digit addition)
        operator_pos = 3
        if head_attentions.shape[1] > operator_pos:
            stats['avg_attention_to_operator'] = head_attentions[:, :, operator_pos].mean().item()
        
        # Average attention to same position (diagonal)
        diagonal_attention = torch.diagonal(head_attentions, dim1=1, dim2=2)
        stats['avg_diagonal_attention'] = diagonal_attention.mean().item()
        
        # Average attention to carry positions (last few positions)
        carry_positions = [-2, -1]  # positions that might involve carry
        carry_attention = []
        for pos in carry_positions:
            if abs(pos) <= head_attentions.shape[1]:
                carry_attention.append(head_attentions[:, :, pos])
        if carry_attention:
            stats['avg_attention_to_carry'] = torch.cat(carry_attention).mean().item()
        
        # Entropy of attention distribution
        # Flatten attention weights and compute entropy
        flat_attentions = head_attentions.view(head_attentions.shape[0], -1)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        flat_attentions = flat_attentions + epsilon
        entropy = -(flat_attentions * torch.log(flat_attentions)).sum(dim=1).mean()
        stats['avg_entropy'] = entropy.item()
        
        # Attention sparsity (fraction of weights above threshold)
        threshold = 0.1
        sparsity = (head_attentions > threshold).float().mean().item()
        stats['sparsity'] = sparsity
        
        head_stats[f'head_{head_idx}'] = stats

    # Save analysis results
    with open(output_dir / 'head_analysis.json', 'w') as f:
        json.dump(head_stats, f, indent=2)

    return head_stats


def ablation_study(model, dataloader, device, output_dir):
    """
    Perform head ablation study.

    Test model performance when individual heads are disabled.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run on
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running head ablation study...")

    # Get baseline accuracy
    baseline_acc = evaluate_model(model, dataloader, device)
    print(f"Baseline accuracy: {baseline_acc:.2%}")

    ablation_results = {'baseline': baseline_acc}

    # For each layer and head:
    num_encoder_layers = len(model.encoder_layers)
    num_decoder_layers = len(model.decoder_layers)
    num_heads = model.encoder_layers[0].self_attn.num_heads
    
    for layer_type in ['encoder', 'decoder']:
        num_layers = num_encoder_layers if layer_type == 'encoder' else num_decoder_layers
        layers = model.encoder_layers if layer_type == 'encoder' else model.decoder_layers
        
        for layer_idx in range(num_layers):
            layer = layers[layer_idx]
            
            # Test self-attention heads
            for head_idx in range(num_heads):
    # 1. Temporarily zero out the head's output
                original_weight = layer.self_attn.W_o.weight.data.clone()
                original_bias = layer.self_attn.W_o.bias.data.clone()
                
                # Zero out the head's contribution in output projection
                head_start = head_idx * layer.self_attn.d_k
                head_end = (head_idx + 1) * layer.self_attn.d_k
                layer.self_attn.W_o.weight.data[:, head_start:head_end] = 0
                
    # 2. Evaluate model performance
                acc = evaluate_model(model, dataloader, device)
                performance_drop = baseline_acc - acc
                
    # 3. Restore the head
                layer.self_attn.W_o.weight.data = original_weight
                layer.self_attn.W_o.bias.data = original_bias
                
    # 4. Record the performance drop
                key = f'{layer_type}_layer_{layer_idx}_self_attn_head_{head_idx}'
                ablation_results[key] = {
                    'accuracy': acc,
                    'performance_drop': performance_drop
                }
            
            # Test cross-attention heads (only for decoder)
            if layer_type == 'decoder':
                for head_idx in range(num_heads):
                    # 1. Temporarily zero out the head's output
                    original_weight = layer.cross_attn.W_o.weight.data.clone()
                    original_bias = layer.cross_attn.W_o.bias.data.clone()
                    
                    # Zero out the head's contribution in output projection
                    head_start = head_idx * layer.cross_attn.d_k
                    head_end = (head_idx + 1) * layer.cross_attn.d_k
                    layer.cross_attn.W_o.weight.data[:, head_start:head_end] = 0
                    
                    # 2. Evaluate model performance
                    acc = evaluate_model(model, dataloader, device)
                    performance_drop = baseline_acc - acc
                    
                    # 3. Restore the head
                    layer.cross_attn.W_o.weight.data = original_weight
                    layer.cross_attn.W_o.bias.data = original_bias
                    
                    # 4. Record the performance drop
                    key = f'{layer_type}_layer_{layer_idx}_cross_attn_head_{head_idx}'
                    ablation_results[key] = {
                        'accuracy': acc,
                        'performance_drop': performance_drop
                    }

    # Save ablation results
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)

    # Create visualization of head importance
    plot_head_importance(ablation_results, output_dir / 'head_importance.png')

    return ablation_results


def evaluate_model(model, dataloader, device):
    """
    Evaluate model accuracy.

    Args:
        model: Model to evaluate
        dataloader: Test dataloader
        device: Device to run on

    Returns:
        Accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            # Generate predictions
            predictions = model.generate(inputs, max_len=targets.size(1))
            
            # Ensure predictions and targets have the same length
            if predictions.size(1) != targets.size(1):
                # Pad or truncate predictions to match targets length
                if predictions.size(1) < targets.size(1):
                    # Pad predictions with padding token (0)
                    pad_length = targets.size(1) - predictions.size(1)
                    padding = torch.zeros(predictions.size(0), pad_length, dtype=predictions.dtype, device=predictions.device)
                    predictions = torch.cat([predictions, padding], dim=1)
                else:
                    # Truncate predictions
                    predictions = predictions[:, :targets.size(1)]
            
            # Compare with targets
            # Create mask for non-padding positions in targets
            mask = (targets != 0)  # True for non-padding positions
            
            # Check if predictions match targets at non-padding positions
            # Only compare where mask is True (non-padding positions)
            matches = (predictions == targets)
            
            # A sequence is correct if all non-padding positions match
            # For padding positions, we don't care what the prediction is
            correct_sequences = (matches | ~mask).all(dim=1)
            
            correct += correct_sequences.sum().item()
            total += targets.size(0)

    return correct / total


def plot_head_importance(ablation_results, save_path):
    """
    Visualize head importance from ablation study.

    Args:
        ablation_results: Dictionary of ablation results
        save_path: Path to save figure
    """
    # Extract performance drops for each head
    baseline = ablation_results['baseline']

    # Filter out baseline and extract head data
    head_data = []
    head_names = []
    
    for key, value in ablation_results.items():
        if key != 'baseline' and 'performance_drop' in value:
            head_data.append(value['performance_drop'])
            head_names.append(key)
    
    if not head_data:
        print("No head ablation data found")
        return

    plt.figure(figsize=(12, 6))

    # Plot bars for each head
    bars = plt.bar(range(len(head_data)), head_data, alpha=0.7)
    
    # Color bars based on performance drop magnitude
    for i, bar in enumerate(bars):
        if head_data[i] > 0.05:  # High importance
            bar.set_color('red')
        elif head_data[i] > 0.01:  # Medium importance
            bar.set_color('orange')
        else:  # Low importance
            bar.set_color('green')

    plt.xlabel('Head')
    plt.ylabel('Accuracy Drop')
    plt.title('Head Importance (Accuracy Drop When Removed)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to free memory and avoid blocking


def visualize_example_predictions(model, dataloader, device, output_dir, num_examples=5):
    """
    Visualize model predictions on example inputs.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        output_dir: Directory to save visualizations
        num_examples: Number of examples to visualize
    """
    output_dir = Path(output_dir)
    (output_dir / 'examples').mkdir(parents=True, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_examples:
                break

            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            # Take first sample from batch
            input_seq = inputs[0:1]
            target_seq = targets[0]

            # Generate prediction
            # TODO: Use model.generate() to get prediction
            prediction = model.generate(input_seq, max_len=target_seq.size(0))

            # Convert to strings for visualization
            input_str = ' '.join(map(str, input_seq[0].cpu().numpy()))
            target_str = ''.join(map(str, target_seq.cpu().numpy()))
            pred_str = ''.join(map(str, prediction[0].cpu().numpy()))

            print(f"\nExample {batch_idx + 1}:")
            print(f"  Input:  {input_str}")
            print(f"  Target: {target_str}")
            print(f"  Pred:   {pred_str}")
            print(f"  Correct: {target_str == pred_str}")

            # TODO: Extract and visualize attention for this example
            # Save attention heatmaps to output_dir / 'examples' / f'example_{batch_idx}.png'
            try:
                # Get attention weights by running forward pass with hooks
                encoder_attentions = []
                decoder_self_attentions = []
                decoder_cross_attentions = []
                
                def make_hook(attention_list):
                    def hook(module, input, output):
                        attention_list.append(output[1].detach().cpu())
                    return hook
                
                # Register hooks
                encoder_hooks = []
                for layer in model.encoder_layers:
                    hook = make_hook(encoder_attentions)
                    encoder_hooks.append(layer.self_attn.register_forward_hook(hook))
                
                # Run forward pass
                decoder_input = target_seq[:-1].unsqueeze(0)
                tgt_mask = create_causal_mask(decoder_input.size(1), device=device)
                outputs = model(input_seq, decoder_input, tgt_mask=tgt_mask)
                
                # Remove hooks
                for hook in encoder_hooks:
                    hook.remove()
                
                # Visualize attention patterns
                if encoder_attentions:
                    # Use the first layer's attention
                    attention_weights = encoder_attentions[0][0]  # [num_heads, seq_len, seq_len]
                    
                    # Create token labels
                    input_tokens = [str(x) for x in input_seq[0].cpu().numpy()]
                    output_tokens = input_tokens  # For self-attention
                    
                    # Save attention heatmap
                    save_path = output_dir / 'examples' / f'example_{batch_idx}_attention.png'
                    visualize_attention_pattern(
                        attention_weights.numpy(),
                        input_tokens,
                        output_tokens,
                        title=f"Example {batch_idx + 1} - Encoder Self-Attention",
                        save_path=save_path
                    )
                    
            except Exception as e:
                print(f"Could not extract attention for example {batch_idx + 1}: {e}")

def generate_attention_patterns(attention_data, model, output_dir, num_samples=10):
    """
    Generate attention pattern heatmaps for each head.
    Each head will have one plot containing all samples as subplots.
    
    Args:
        attention_data: Dictionary with attention weights
        model: Trained model
        output_dir: Output directory
        num_samples: Number of samples to visualize
    """
    num_samples = min(num_samples, len(attention_data['inputs']))
    
    print(f"Generating attention patterns for {num_samples} samples...")
    
    # Collect all attention weights and tokens for each head
    encoder_heads_data = {}  # {head_idx: [(attention_matrix, input_tokens, sample_idx), ...]}
    decoder_cross_heads_data = {}  # {head_idx: [(attention_matrix, input_tokens, output_tokens, sample_idx), ...]}
    
    # First pass: collect all data
    for sample_idx in range(num_samples):
        try:
            input_seq = attention_data['inputs'][sample_idx]
            target_seq = attention_data['targets'][sample_idx]
            
            # Convert to numpy array if needed
            if isinstance(input_seq, torch.Tensor):
                input_seq = input_seq.cpu().numpy()
            if isinstance(target_seq, torch.Tensor):
                target_seq = target_seq.cpu().numpy()
            
            # Ensure they are 1D arrays
            if len(input_seq.shape) > 1:
                input_seq = input_seq.flatten()
            if len(target_seq.shape) > 1:
                target_seq = target_seq.flatten()
            
            # Prepare tokens
            input_tokens = []
            for tok in input_seq:
                if tok == 0:  # padding
                    break
                input_tokens.append('+' if tok == 10 else str(int(tok)))
            
            # For decoder cross-attention:
            # decoder_input = targets[:, :-1], so decoder positions correspond to target_seq[1:]
            # Extract output_tokens matching decoder positions (skip first token of target_seq)
            output_tokens_decoder = []
            for tok in target_seq[1:]:  # Skip first token to match decoder_input
                if tok == 0:
                    break
                output_tokens_decoder.append(str(int(tok)))
            
            # Keep original output_tokens for reference (full sequence)
            output_tokens_full = []
            for tok in target_seq:
                if tok == 0:
                    break
                output_tokens_full.append(str(int(tok)))
            
            # Process encoder attention
            if attention_data['encoder_attention']:
                encoder_attns = attention_data['encoder_attention']
                
                if sample_idx < len(encoder_attns):
                    sample_encoder_attns = encoder_attns[sample_idx]
                    layer_attentions = []
                    for layer_attn in sample_encoder_attns:
                        if layer_attn is not None:
                            if not isinstance(layer_attn, torch.Tensor):
                                if isinstance(layer_attn, (list, np.ndarray)):
                                    layer_attn = torch.tensor(layer_attn)
                                else:
                                    continue
                            layer_attentions.append(layer_attn)
                    
                    if layer_attentions:
                        avg_encoder_attn = torch.stack(layer_attentions).mean(dim=0)
                        if isinstance(avg_encoder_attn, torch.Tensor):
                            avg_encoder_attn = avg_encoder_attn.numpy()
                        
                        # Collect data for each head
                        for head_idx in range(avg_encoder_attn.shape[0]):
                            single_head = avg_encoder_attn[head_idx]
                            
                            if isinstance(single_head, torch.Tensor):
                                single_head = single_head.detach().cpu().numpy()
                            
                            while len(single_head.shape) > 2:
                                single_head = single_head.squeeze()
                            
                            if len(single_head.shape) != 2:
                                continue
                            
                            if head_idx not in encoder_heads_data:
                                encoder_heads_data[head_idx] = []
                            encoder_heads_data[head_idx].append((single_head, input_tokens, sample_idx))
            
            # Process decoder cross-attention
            if attention_data['decoder_cross_attention']:
                cross_attns = attention_data['decoder_cross_attention']
                
                if sample_idx < len(cross_attns):
                    sample_cross_attns = cross_attns[sample_idx]
                    layer_attentions = []
                    for layer_attn in sample_cross_attns:
                        if layer_attn is not None:
                            if not isinstance(layer_attn, torch.Tensor):
                                if isinstance(layer_attn, (list, np.ndarray)):
                                    layer_attn = torch.tensor(layer_attn)
                                else:
                                    continue
                            layer_attentions.append(layer_attn)
                    
                    if layer_attentions:
                        avg_cross_attn = torch.stack(layer_attentions).mean(dim=0)
                        if isinstance(avg_cross_attn, torch.Tensor):
                            avg_cross_attn = avg_cross_attn.numpy()
                        
                        # Collect data for each head
                        for head_idx in range(avg_cross_attn.shape[0]):
                            single_head = avg_cross_attn[head_idx]
                            
                            if isinstance(single_head, torch.Tensor):
                                single_head = single_head.detach().cpu().numpy()
                            
                            while len(single_head.shape) > 2:
                                single_head = single_head.squeeze()
                            
                            if len(single_head.shape) != 2:
                                continue
                            
                            # Ensure attention matrix dimensions match tokens
                            # Decoder cross-attention: [decoder_len, encoder_len]
                            # decoder_len should match len(output_tokens_decoder)
                            # encoder_len should match len(input_tokens)
                            
                            # Trim attention matrix if needed to match token lengths
                            decoder_len = single_head.shape[0]
                            encoder_len = single_head.shape[1]
                            
                            # Adjust if dimensions don't match
                            if decoder_len != len(output_tokens_decoder):
                                # Trim or pad to match
                                if decoder_len < len(output_tokens_decoder):
                                    output_tokens_decoder = output_tokens_decoder[:decoder_len]
                                else:
                                    # Pad attention matrix if needed (shouldn't happen)
                                    pass
                            
                            if encoder_len != len(input_tokens):
                                # Trim or pad to match
                                if encoder_len < len(input_tokens):
                                    input_tokens_trimmed = input_tokens[:encoder_len]
                                else:
                                    input_tokens_trimmed = input_tokens + [''] * (encoder_len - len(input_tokens))
                            else:
                                input_tokens_trimmed = input_tokens
                            
                            if head_idx not in decoder_cross_heads_data:
                                decoder_cross_heads_data[head_idx] = []
                            decoder_cross_heads_data[head_idx].append(
                                (single_head, input_tokens_trimmed, output_tokens_decoder, sample_idx)
                            )
        
        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            continue
    
    # Second pass: create combined plots for each head
    # Generate encoder attention plots
    for head_idx, samples_data in encoder_heads_data.items():
        if not samples_data:
            continue
        
        # Create subplot grid: arrange samples in a grid
        num_samples_for_head = len(samples_data)
        cols = min(5, num_samples_for_head)  # 5 columns
        rows = (num_samples_for_head + cols - 1) // cols  # Calculate rows needed
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        if num_samples_for_head == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1) if cols > 1 else axes.reshape(-1, 1)
        axes = axes.flatten()
        
        for subplot_idx, (attn_matrix, tokens, sample_idx) in enumerate(samples_data):
            if subplot_idx >= len(axes):
                break
            
            ax = axes[subplot_idx]
            
            # Plot heatmap
            sns.heatmap(
                attn_matrix,
                ax=ax,
                cmap='Blues',
                cbar=True,
                square=True,
                xticklabels=tokens,
                yticklabels=tokens,
                vmin=0,
                vmax=1
            )
            
            ax.set_title(f'Sample {sample_idx}', fontsize=10)
            ax.set_xlabel('Input Position', fontsize=8)
            ax.set_ylabel('Input Position', fontsize=8)
        
        # Hide unused subplots
        for idx in range(num_samples_for_head, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Encoder Head {head_idx + 1} - All Samples', fontsize=14, y=0.995)
        plt.tight_layout()
        
        save_path = output_dir / 'attention_patterns' / f'encoder_head_{head_idx+1}_all_samples.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Generate decoder cross-attention plots
    for head_idx, samples_data in decoder_cross_heads_data.items():
        if not samples_data:
            continue
        
        # Create subplot grid
        num_samples_for_head = len(samples_data)
        cols = min(5, num_samples_for_head)
        rows = (num_samples_for_head + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        if num_samples_for_head == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1) if cols > 1 else axes.reshape(-1, 1)
        axes = axes.flatten()
        
        for subplot_idx, (attn_matrix, input_tokens, output_tokens, sample_idx) in enumerate(samples_data):
            if subplot_idx >= len(axes):
                break
            
            ax = axes[subplot_idx]
            
            # Ensure dimensions match
            decoder_len, encoder_len = attn_matrix.shape
            
            # Trim tokens if needed
            input_tokens_plot = input_tokens[:encoder_len] if len(input_tokens) > encoder_len else input_tokens
            output_tokens_plot = output_tokens[:decoder_len] if len(output_tokens) > decoder_len else output_tokens
            
            # Plot heatmap
            sns.heatmap(
                attn_matrix,
                ax=ax,
                cmap='Blues',
                cbar=True,
                square=True,
                xticklabels=input_tokens_plot,
                yticklabels=output_tokens_plot,
                vmin=0,
                vmax=1
            )
            
            ax.set_title(f'Sample {sample_idx}', fontsize=10)
            ax.set_xlabel('Input Position', fontsize=8)
            ax.set_ylabel('Output Position', fontsize=8)
        
        # Hide unused subplots
        for idx in range(num_samples_for_head, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Decoder Cross-Attention Head {head_idx + 1} - All Samples', fontsize=14, y=0.995)
        plt.tight_layout()
        
        save_path = output_dir / 'attention_patterns' / f'decoder_cross_head_{head_idx+1}_all_samples.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Generated {len(encoder_heads_data)} encoder head plots and {len(decoder_cross_heads_data)} decoder cross-attention head plots")

def main():
    parser = argparse.ArgumentParser(description='Analyze attention patterns')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Load model
    vocab_size = get_vocab_size()
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512
    ).to(args.device)

    model.load_state_dict(torch.load(args.model_path))
    print(f"Loaded model from {args.model_path}")

    # Load data
    _, _, test_loader = create_dataloaders(args.data_dir, args.batch_size)

    # Create output directories
    output_dir = Path(args.output_dir)
    (output_dir / 'attention_patterns').mkdir(parents=True, exist_ok=True)
    (output_dir / 'head_analysis').mkdir(parents=True, exist_ok=True)

    # Extract attention weights
    print("Extracting attention weights...")
    attention_data = extract_attention_weights(
        model, test_loader, args.device, args.num_samples
    )

    # generate attention pattern visualization
    print("Generating attention pattern heatmaps...")
    generate_attention_patterns(attention_data, model, output_dir)

    # Analyze head specialization
    head_stats = analyze_head_specialization(
        attention_data, output_dir / 'head_analysis'
    )

    # Run ablation study
    ablation_results = ablation_study(
        model, test_loader, args.device, output_dir / 'head_analysis'
    )

    # Visualize example predictions
    visualize_example_predictions(
        model, test_loader, args.device, output_dir, num_examples=5
    )

    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()