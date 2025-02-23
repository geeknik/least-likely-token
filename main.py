import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.distributions import Categorical

# Initialize models and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token for padding

model_M = GPT2LMHeadModel.from_pretrained("gpt2")  # Standard model (fixed)
model_N = GPT2LMHeadModel.from_pretrained("gpt2")  # Least likely predictor (trainable)
optimizer_N = optim.Adam(model_N.parameters(), lr=5e-5)  # Increased learning rate from 1e-5 to 5e-5

# Set model_M to evaluation mode (no training)
model_M.eval()

# Small epsilon to prevent log(0)
epsilon = 1e-10

# Moving average baseline for variance reduction
global_baseline = 0.0
alpha = 0.01  # Smoothing factor for moving average (1% new data, 99% old)

def train_least_likely_predictor(train_sequences, val_sequences, num_epochs=50, batch_size=2):
    """
    Train Model N to predict the least likely next token using reinforcement learning.
    
    Args:
        train_sequences (list): List of training sequences (strings).
        val_sequences (list): List of validation sequences (strings).
        num_epochs (int): Number of training epochs (increased to 50).
        batch_size (int): Number of sequences per batch.
    """
    global global_baseline
    for epoch in range(num_epochs):
        model_N.train()
        total_loss = 0
        num_batches = max(1, len(train_sequences) // batch_size)
        sampled_probs = []
        
        for i in range(0, len(train_sequences), batch_size):
            batch = train_sequences[i:i + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask
            
            # Get output distribution from Model N
            outputs_N = model_N(input_ids, attention_mask=attention_mask)
            logits_N = outputs_N.logits[:, -1, :]  # Logits for next token
            probs_N = nn.functional.softmax(logits_N, dim=-1)
            
            # Sample tokens from N's distribution
            m = Categorical(probs_N)
            sampled_tokens = m.sample()
            
            # Compute P_M(t|s) for sampled tokens
            with torch.no_grad():
                outputs_M = model_M(input_ids, attention_mask=attention_mask)
                logits_M = outputs_M.logits[:, -1, :]
                probs_M = nn.functional.softmax(logits_M, dim=-1)
                P_M_t_given_s = probs_M[torch.arange(len(batch)), sampled_tokens]
            
            # Compute rewards: -log(P_M(t|s) + epsilon)
            rewards = -torch.log(P_M_t_given_s + epsilon)
            
            # Use moving average baseline instead of batch mean
            batch_mean_reward = rewards.mean().item()
            advantage = rewards - global_baseline
            global_baseline = (1 - alpha) * global_baseline + alpha * batch_mean_reward
            
            # Compute policy gradient loss: -log(N(t|s)) * advantage
            log_probs = m.log_prob(sampled_tokens)
            loss = -torch.mean(log_probs * advantage)
            total_loss += loss.item()
            
            # Update Model N
            optimizer_N.zero_grad()
            loss.backward()
            optimizer_N.step()
            
            # Collect sampled probabilities and log example
            sampled_probs.extend(P_M_t_given_s.tolist())
            if i == 0:  # Log first batch for visibility
                sampled_token = tokenizer.decode(sampled_tokens[0])
                print(f"Epoch {epoch + 1}, Sampled: '{sampled_token}', P_M(t|s): {P_M_t_given_s[0].item():.6f}")
        
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
            encoded = tokenizer(seq, return_tensors="pt")
            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask
            outputs_N = model_N(input_ids, attention_mask=attention_mask)
            logits_N = outputs_N.logits[:, -1, :]
            probs_N = nn.functional.softmax(logits_N, dim=-1)
            t_pred = torch.argmax(probs_N, dim=-1).item()
            
            outputs_M = model_M(input_ids, attention_mask=attention_mask)
            logits_M = outputs_M.logits[:, -1, :]
            probs_M = nn.functional.softmax(logits_M, dim=-1)
            P_M_t_pred = probs_M[0, t_pred].item()
            total_prob += P_M_t_pred
            
            # Log predicted token and its probability
            predicted_token = tokenizer.decode(t_pred)
            print(f"Validation Seq: '{seq}', Predicted: '{predicted_token}', P_M(t|s): {P_M_t_pred:.6f}")
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
        encoded = tokenizer(sequence, return_tensors="pt")
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        outputs_N = model_N(input_ids, attention_mask=attention_mask)
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
    train_least_likely_predictor(train_sequences, val_sequences, num_epochs=50, batch_size=2)
    
    test_sequence = "The cat sat on the"
    predicted_token = predict_least_likely_token(test_sequence)
    print(f"\nInput: '{test_sequence}'")
    print(f"Predicted least likely token: '{predicted_token}'")
