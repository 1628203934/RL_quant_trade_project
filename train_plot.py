from model_libraries import plt
from implement_RL_model import agent


def plot_history(agent):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(agent.history['loss'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Reward', color=color)

    # Normalize rewards for better visualization
    normalized_rewards = [r / max(abs(r) for r in agent.history['reward']) for r in agent.history['reward']]

    ax2.plot(normalized_rewards, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

plot_history(agent)