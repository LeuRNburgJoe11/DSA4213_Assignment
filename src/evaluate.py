# src/evaluate.py
import math
import torch

def compute_perplexity(avg_cross_entropy_bits):
    """Converts average cross-entropy (in bits) to perplexity."""
    return 2 ** avg_cross_entropy_bits


def evaluate_neural_model(model, data_loader, criterion, device):
    """
    Evaluates a PyTorch Neural LM (RNN, LSTM, Transformer).
    Excludes padding tokens from average loss calculation.
    """
    model.eval()
    total_loss_nats = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass: outputs shape [batch_size, seq_len, vocab_size]
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Flatten tensors for loss calculation
            vocab_size = outputs.size(-1)
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)
            
            # Loss with reduction='sum' and ignore_index for <pad>
            loss = criterion(outputs, targets) 
            
            # Count only non-padding tokens
            non_pad_tokens = (targets != criterion.ignore_index).sum().item()
            
            total_loss_nats += loss.item()
            total_tokens += non_pad_tokens

    if total_tokens == 0:
        return float('inf'), float('inf')

    # Convert PyTorch's natural log loss (nats) to base-2 log loss (bits)
    avg_cross_entropy_bits = (total_loss_nats / total_tokens) / math.log(2)
    ppl = compute_perplexity(avg_cross_entropy_bits)
    
    return avg_cross_entropy_bits, ppl