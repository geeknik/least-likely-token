import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.distributions import Categorical

# Initialize models and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model_M = GPT2LMHeadModel.from_pretrained("gpt2")  # Standard model (fixed)
model_N = GPT2LMHeadModel.from_pretrained("gpt2")  # Least likely predictor (trainable)
optimizer_N = optim.Adam(model_N.parameters(), lr=1e-5)

# Set model_M to evaluation mode (no training)
model_M.eval()

# Small epsilon to prevent log(0)
epsilon = 1e-10

def train_least_likely_predictor(train_sequences, val_sequences, num_epochs=10, batch_size=1):
    """
    Train Model N to predict the least likely next token using reinforcement learning.
    
    Args:
        train_sequences (list): List of training sequences (strings).
        val_sequences (list): List of validation sequences (strings).
        num_epochs (int): Number of training epochs.
        batch_size (int): Number of sequences per batch.
    """
    for epoch in range(num_epochs):
        model_N.train()
        total_loss = 0
        num_batches = max(1, len(train_sequences) // batch_size)
        sampled_probs = []
        
        for i in range(0, len(train_sequences), batch_size):
            batch = train_sequences[i:i + batch_size]
            input_ids = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).input_ids
            
            # Get output distribution from Model N
            outputs_N = model_N(input_ids)
            logits_N = outputs_N.logits[:, -1, :]  # Logits for next token
            probs_N = nn.functional.softmax(logits_N, dim=-1)
            
            # Sample tokens from N's distribution
            m = Categorical(probs_N)
            sampled_tokens = m.sample()
            
            # Compute P_M(t|s) for sampled tokens
            with torch.no_grad():
                outputs_M = model_M(input_ids)
                logits_M = outputs_M.logits[:, -1, :]
                probs_M = nn.functional.softmax(logits_M, dim=-1)
                P_M_t_given_s = probs_M[torch.arange(len(batch)), sampled_tokens]
            
            # Compute rewards: -log(P_M(t|s) + epsilon)
            rewards = -torch.log(P_M_t_given_s + epsilon)
            
            # Compute baseline and advantage
            baseline = rewards.mean()
            advantage = rewards - baseline
            
            # Compute policy gradient loss: -log(N(t|s)) * advantage
            log_probs = m.log_prob(sampled_tokens)
            loss = -torch.mean(log_probs * advantage)
            total_loss += loss.item()
            
            # Update Model N
            optimizer_N.zero_grad()
            loss.backward()
            optimizer_N.step()
            
            # Collect sampled probabilities
            sampled_probs.extend(P_M_t_given_s.tolist())
        
        # Report training metrics
        avg_loss = total_loss / num_batches
        avg_sampled_prob = sum(sampled_probs) / len(sampled_probs)
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Avg Sampled P_M(t|s): {avg_sampled_prob:.6f}")
        
        # Validate
        val_avg_prob = validate(model_N, model_M, val_sequences)
        print(f"Validation Avg P_M(t|s): {val_avg_prob:.6f}")

def validate(model_N, model_M, val_sequences):
    """
    Evaluate Model N on validation sequences.
    
    Args:
        model_N: Least likely predictor model.
        model_M: Standard language model.
        val_sequences (list): Validation sequences.
    
    Returns:
        float: Average P_M(t|s) of predicted tokens.
    """
    model_N.eval()
    total_prob = 0
    with torch.no_grad():
        for seq in val_sequences:
            input_ids = tokenizer.encode(seq, return_tensors="pt")
            outputs_N = model_N(input_ids)
            logits_N = outputs_N.logits[:, -1, :]
            probs_N = nn.functional.softmax(logits_N, dim=-1)
            t_pred = torch.argmax(probs_N, dim=-1).item()
            
            outputs_M = model_M(input_ids)
            logits_M = outputs_M.logits[:, -1, :]
            probs_M = nn.functional.softmax(logits_M, dim=-1)
            P_M_t_pred = probs_M[0, t_pred].item()
            total_prob += P_M_t_pred
    avg_prob = total_prob / len(val_sequences)
    model_N.train()
    return avg_prob

def predict_least_likely_token(sequence):
    """
    Predict the least likely next token for a sequence.
    
    Args:
        sequence (str): Input sequence.
    
    Returns:
        str: Predicted token.
    """
    model_N.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(sequence, return_tensors="pt")
        outputs_N = model_N(input_ids)
        logits_N = outputs_N.logits[:, -1, :]
        probs_N = nn.functional.softmax(logits_N, dim=-1)
        t_pred = torch.argmax(probs_N, dim=-1).item()
    model_N.train()
    return tokenizer.decode(t_pred)

# Example usage
if __name__ == "__main__":
    train_sequences = [
        "The cat sat on the",
        "I like to eat",
        "Once upon a time",
        "The dog ran across"
    ]
    val_sequences = [
        "In a galaxy far",
        "The quick brown fox"
    ]
    
    print("Training the apex predator of least likely predictors...")
    train_least_likely_predictor(train_sequences, val_sequences, num_epochs=5, batch_size=2)
    
    test_sequence = "The cat sat on the"
    predicted_token = predict_least_likely_token(test_sequence)
    print(f"\nInput: '{test_sequence}'")
    print(f"Predicted least likely token: '{predicted_token}'")
