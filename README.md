# RL Sentiment Agent 🎯

**Reinforcement Learning agent that learns optimal sentiment classification through trial and error**

## 🎯 Why Reinforcement Learning?

Shows you understand:
- **Learning from feedback** (not just supervised)
- **Exploration vs exploitation** (RL fundamentals)
- **Q-learning / Policy gradients** (RL algorithms)
- **Adaptive systems** (learns and improves)

## 🧩 Atomic Concept

**Core AI Principle**: Reinforcement learning for decision-making under uncertainty

## 🏗️ How It Works

### Traditional Sentiment Classifier
```python
# Static: Trained once, never changes
model.train(labeled_data)
prediction = model.predict(text)  # Fixed behavior
```

### RL Sentiment Agent
```python
# Dynamic: Learns from every interaction
agent = RLSentimentAgent()

# Start with random policy
prediction = agent.predict(text)

# Get feedback (reward)
user_feedback = get_user_rating(prediction)

# LEARN and improve
agent.learn(text, prediction, user_feedback)

# Next time: Better predictions!
```

## 🔬 Architecture

```
┌─────────────────────────────────────┐
│      RL Sentiment Agent              │
├─────────────────────────────────────┤
│  STATE        │ Text features         │
│               │ (embeddings)          │
├───────────────┼───────────────────────┤
│  ACTION       │ Predict sentiment     │
│               │ (pos/neg/neutral)     │
├───────────────┼───────────────────────┤
│  REWARD       │ User feedback         │
│               │ (+1 correct, -1 wrong)│
├───────────────┼───────────────────────┤
│  POLICY       │ Q-table or Neural Net │
│               │ (learns over time)    │
└───────────────┴───────────────────────┘
```

## 💡 Implementation: Q-Learning

```python
class RLSentimentAgent:
    """Q-Learning agent for sentiment classification"""
    
    def __init__(self):
        # Q-table: Q(state, action) = expected reward
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # Hyperparameters
        self.alpha = 0.1    # Learning rate
        self.gamma = 0.9    # Discount factor
        self.epsilon = 0.1  # Exploration rate
    
    def predict(self, text):
        """Choose action (sentiment)"""
        state = self._encode_state(text)
        
        # Exploration vs Exploitation
        if random.random() < self.epsilon:
            # EXPLORE: Try random action
            return random.choice(['positive', 'negative', 'neutral'])
        else:
            # EXPLOIT: Use learned policy
            return max(self.actions, 
                      key=lambda a: self.Q[state][a])
    
    def learn(self, text, action, reward, next_text=None):
        """Q-learning update (Bellman equation)"""
        state = self._encode_state(text)
        next_state = self._encode_state(next_text) if next_text else state
        
        # Current Q-value
        old_value = self.Q[state][action]
        
        # Best next Q-value
        next_max = max(self.Q[next_state].values()) if next_state else 0
        
        # Q-learning update
        new_value = old_value + self.alpha * (
            reward + self.gamma * next_max - old_value
        )
        
        self.Q[state][action] = new_value
    
    def _encode_state(self, text):
        """Convert text to state representation"""
        # Simple: Use word embeddings or TF-IDF
        return hash(text) % 10000  # Discretize for Q-table
```

## 💡 Alternative: Policy Gradient

```python
class PolicyGradientAgent(nn.Module):
    """Neural network policy for sentiment"""
    
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, hidden_dim)
        self.policy = nn.Linear(hidden_dim, 3)  # 3 sentiments
        
        self.saved_log_probs = []
        self.rewards = []
    
    def forward(self, text):
        """Forward pass: output action probabilities"""
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        action_probs = F.softmax(self.policy(lstm_out[-1]), dim=-1)
        return action_probs
    
    def select_action(self, text):
        """Sample action from policy"""
        probs = self.forward(text)
        action = torch.multinomial(probs, 1)
        
        # Save for training
        self.saved_log_probs.append(torch.log(probs[action]))
        return action.item()
    
    def learn(self):
        """REINFORCE algorithm"""
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        
        # Policy gradient
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Backprop
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        
        # Clear buffers
        self.rewards.clear()
        self.saved_log_probs.clear()
```

## 📁 Project Structure

```
rl-sentiment-agent/
├── agent/
│   ├── q_learning.py
│   ├── policy_gradient.py
│   └── dqn.py
├── environments/
│   ├── sentiment_env.py
│   └── reward_shaping.py
├── training/
│   ├── train_q_learning.py
│   └── train_pg.py
├── evaluation/
│   ├── metrics.py
│   └── visualize_learning.py
├── notebooks/
│   ├── q_learning_demo.ipynb
│   └── policy_gradient_demo.ipynb
└── tests/
    └── test_agent.py
```

## 🎓 What You'll Learn

1. **Q-Learning Algorithm**
2. **Policy Gradient Methods**
3. **Exploration vs Exploitation**
4. **Reward Shaping**
5. **Markov Decision Process**
6. **RL Evaluation Metrics**

## 📊 Learning Curve

```
Cumulative Reward Over Time

    │                              ╱─────
800 │                         ╱────
    │                    ╱────
600 │               ╱────
    │          ╱────
400 │     ╱────
    │ ────
200 │
    └─────────────────────────────────────
    0   100  200  300  400  500  Episodes
```

The agent:
- Starts with random guesses
- Explores different strategies
- Learns from feedback
- Converges to optimal policy

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Train Q-learning agent
python -m training.train_q_learning \
  --episodes 1000 \
  --learning-rate 0.1

# Train policy gradient agent
python -m training.train_pg \
  --episodes 500 \
  --batch-size 32

# Evaluate
python -m evaluation.metrics \
  --checkpoint best_agent.pth

# Visualize learning
python -m evaluation.visualize_learning
```

## 🎯 Use Cases

1. **Adaptive Sentiment Analysis**: Learns user preferences
2. **Online Learning**: Improves with each interaction
3. **Personalized Classification**: Different users, different policies
4. **Active Learning**: Queries uncertain cases

## 📊 Comparison

| Approach | Training Data | Adaptation | User Feedback |
|----------|---------------|------------|---------------|
| **Supervised ML** | Large labeled set | No | Not used |
| **This RL Agent** | Learns from scratch | Yes | Core mechanism |

## 🎓 Interview Value: ⭐⭐⭐⭐⭐

Shows you understand:
- Reinforcement learning fundamentals
- Q-learning and policy gradients
- Exploration strategies
- Online learning systems

## 📚 Resources

- [Sutton & Barto: RL Book](http://incompleteideas.net/book/the-book.html)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Deep RL Course (Hugging Face)](https://huggingface.co/deep-rl-course/unit0/introduction)

## 📄 License

MIT
