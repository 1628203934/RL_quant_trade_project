from model_libraries import deque, np, random
from RL_model import create_dqn_model
from RL_environment import train_env, test_env, train_data, test_data


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = create_dqn_model((state_size,), action_size)
        self.history = {'loss': [], 'reward': []}

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            self.history['loss'].append(history.history['loss'][0])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_history(self, reward):
        self.history['reward'].append(reward)

def train_dqn_agent(agent, env, episodes, batch_size):
    for e in range(episodes):
        print(f'Episode {e}')
        state = env.reset()
        state = np.reshape(state, [1, len(env.data.columns)])
        for time in range(20):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, len(env.data.columns)])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e}/{episodes}, Reward: {reward}, Epsilon: {agent.epsilon}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

def validate_dqn_agent(agent, env):
    state = env.reset()
    state = np.reshape(state, [1, len(env.data.columns)])
    test_rewards = []
    for time in range(len(env.data) - 1):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, len(env.data.columns)])
        state = next_state
        test_rewards.append(reward)
        if done:
            break

    cumulative_return = env.balance + env.shares_held * env.state[0] - env.initial_balance
    annualized_return = (cumulative_return / env.initial_balance) ** (252 / len(env.data)) - 1
    sharpe_ratio = np.mean(test_rewards) / np.std(test_rewards) * np.sqrt(252)
    max_drawdown = np.min(test_rewards) / env.initial_balance
    volatility = np.std(test_rewards) * np.sqrt(252)

    print(f"Cumulative Return: {cumulative_return}")
    print(f"Annualized Return: {annualized_return}")
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Max Drawdown: {max_drawdown}")
    print(f"Volatility: {volatility}")

# DQN Agent
agent = DQNAgent(state_size=len(train_data.columns), action_size=3)

# Train the agent
train_dqn_agent(agent, train_env, episodes=10, batch_size=16)

# Validate the agent
validate_dqn_agent(agent, test_env)


