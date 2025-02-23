# Least Likely Next Token Predictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a **least likely next token predictor** using reinforcement learning. It leverages a pre-trained language model (GPT-2) to estimate token probabilities and trains a second model to predict tokens with the lowest probabilities, making it the "apex predator" of unlikely token predictors.

![image](https://github.com/user-attachments/assets/e96a5cdc-665f-4b7c-bafe-a5c2ee4d0af9)


## Installation

To run this project, ensure you have Python installed. Then, install the required dependencies:
```bash
pip install torch transformers
```
torch: For tensor operations and model training.
transformers: For pre-trained GPT-2 models and tokenization.

## Usage
To train the least likely next token predictor, run the following command:
```bash
python main.py
```
This will train the model using the default sequences provided in the script. You can modify the train_sequences and val_sequences lists in the script to use your own data.

## Example
After training, you can use the model to predict the least likely next token for a given sequence:
```python
test_sequence = "The cat sat on the"
predicted_token = predict_least_likely_token(test_sequence)
print(f"Predicted least likely next token: '{predicted_token}'")
```

## How It Works

The predictor is trained using reinforcement learning with the following components:

- **Standard Language Model (M)**: A pre-trained GPT-2 model that provides the probability distribution $P_M(t|s)$ of the next token $t$ given a sequence $s$. This model remains fixed.
- **Least Likely Predictor (N)**: Another GPT-2 model trained to predict tokens with low probabilities according to model M.

### Training Process

1. **Sampling**: For each input sequence, model N samples a next token from its predicted distribution.
2. **Reward Calculation**: The reward is computed as $-\log(P_M(t|s) + \epsilon)$, where $\epsilon = 10^{-10}$ prevents $\log(0)$. This reward is higher for tokens that are less likely under model M.
3. **Policy Gradient Update**: Using the REINFORCE algorithm with a baseline (mean reward), model N is updated to maximize the expected reward, encouraging it to predict unlikely tokens.

This approach ensures that model N learns to consistently predict tokens that are improbable according to the standard language model.

## Enhancements

- **Optimized Reward Function**: The reward $-\log(P_M(t|s) + \epsilon)$ provides a stronger gradient signal for very unlikely tokens, improving learning efficiency.
- **Variance Reduction**: A baseline (mean reward per batch) is used in the policy gradient to stabilize training by reducing variance in the gradient estimates.
- **Validation**: The average $P_M(t|s)$ of predicted tokens is monitored on a validation set after each epoch to ensure the model generalizes well and avoids overfitting.

These enhancements make this implementation robust, efficient, and highly effective at predicting the least likely tokens.

## Project Structure
main.py: The main script containing the training and inference code.  

## License
This project is licensed under the MIT License - see the LICENSE file for details.
