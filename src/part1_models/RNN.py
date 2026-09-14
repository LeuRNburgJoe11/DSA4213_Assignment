import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src import evaluate
from src.tokenizer import tokenize


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

class VanillaRNNLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: [batch_size, seq_len]
        embeds = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        out, hidden = self.rnn(embeds, hidden)  # out: [batch_size, seq_len, hidden_dim]
        logits = self.fc(out)       # [batch_size, seq_len, vocab_size]
        return logits, hidden


def read_tokens(path):
    """Read and tokenize one text split."""
    text = Path(path).read_text(encoding="utf-8")
    return tokenize(text)


def build_vocabulary(tokens):
    """Build a vocabulary using training tokens only."""
    vocabulary = [PAD_TOKEN, UNK_TOKEN]
    vocabulary.extend(sorted(set(tokens) - {PAD_TOKEN, UNK_TOKEN}))
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    return token_to_id, vocabulary


def make_data_loader(tokens, token_to_id, sequence_length, batch_size, shuffle):
    """Create fixed-length next-token prediction examples."""
    unknown_id = token_to_id[UNK_TOKEN]
    ids = [token_to_id.get(token, unknown_id) for token in tokens]
    inputs = []
    targets = []

    for start in range(0, len(ids) - sequence_length, sequence_length):
        window = ids[start : start + sequence_length + 1]
        if len(window) == sequence_length + 1:
            inputs.append(window[:-1])
            targets.append(window[1:])

    if not inputs:
        raise ValueError("The split is too short for the selected sequence length.")

    dataset = TensorDataset(
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    """Train the model and report validation cross-entropy/perplexity per epoch."""
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

        train_loss = total_loss / total_tokens
        valid_bits, valid_ppl = evaluate.evaluate_neural_model(
            model, valid_loader, criterion, device
        )
        print(
            f"Epoch {epoch:02d} | train loss: {train_loss:.4f} nats/token | "
            f"valid CE: {valid_bits:.4f} bits/token | valid PPL: {valid_ppl:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Train a vanilla RNN language model.")
    parser.add_argument("--train", default="data/processed/train.txt")
    parser.add_argument("--valid", default="data/processed/valid.txt")
    parser.add_argument("--checkpoint", default="results/rnn.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    train_tokens = read_tokens(args.train)
    valid_tokens = read_tokens(args.valid)
    token_to_id, vocabulary = build_vocabulary(train_tokens)
    train_loader = make_data_loader(
        train_tokens, token_to_id, args.sequence_length, args.batch_size, shuffle=True
    )
    valid_loader = make_data_loader(
        valid_tokens, token_to_id, args.sequence_length, args.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VanillaRNNLM(len(vocabulary), args.embed_dim, args.hidden_dim)
    criterion = nn.CrossEntropyLoss(ignore_index=token_to_id[PAD_TOKEN], reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Vocabulary: {len(vocabulary):,} tokens | device: {device}")
    train_model(model, train_loader, valid_loader, criterion, optimizer, device, args.epochs)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "token_to_id": token_to_id,
            "vocabulary": vocabulary,
            "model_config": {
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
            },
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()