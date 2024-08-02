from model_libraries import gym, spaces, np
from data_collect import train_data, test_data


class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()

        self.data = data
        self.n_steps = len(data)
        self.current_step = 0
        self.done = False

        # Action space: 0 - hold, 1 - buy, 2 - sell
        self.action_space = spaces.Discrete(3)

        # Observation space: additional features like RSI, MACD, etc.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(data.columns),), dtype=np.float32)

        # Initial state
        self.state = self.data.iloc[0].values  # All features

        # Initial portfolio value
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0

    def reset(self):
        self.current_step = 0
        self.done = False
        self.balance = self.initial_balance
        self.shares_held = 0
        self.state = self.data.iloc[0].values
        return self.state

    def step(self, action):
        if self.current_step >= self.n_steps - 1:
            self.done = True

        current_price = self.data.iloc[self.current_step]['Close']

        if action == 1:  # Buy
            self.shares_held += 1
            self.balance -= current_price
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price

        self.current_step += 1

        if self.current_step >= self.n_steps:
            self.done = True

        self.state = self.data.iloc[self.current_step].values if not self.done else self.state

        portfolio_value = self.balance + self.shares_held * current_price
        reward = portfolio_value - self.initial_balance

        return self.state, reward, self.done, {}


train_env = TradingEnv(train_data)
test_env = TradingEnv(test_data)
