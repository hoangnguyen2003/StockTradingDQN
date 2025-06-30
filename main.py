from data.data_loader import download_data, preprocess_data
from environment.trading_env import TradingEnvironment
from model.agent import DQNAgent
from config import config

def train_agent():
    raw_data = download_data(config["symbol"], config["start_date"], config["end_date"])
    data = preprocess_data(raw_data)
    
    env = TradingEnvironment(data, config["initial_balance"])
    agent = DQNAgent(config)
    
    for episode in range(config["training"]["episodes"]):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state, config["actions"])
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.replay(config["training"]["batch_size"])
        print(f"Episode {episode+1}/{config['training']['episodes']}, total reward: {total_reward}")

    print("Training complete!")
    return agent, data

def test_agent(agent, data):
    test_env = TradingEnvironment(data, config["initial_balance"])
    state = test_env.reset()
    done = False

    while not done:
        action = agent.act(state, config["actions"])
        next_state, reward, done, _ = test_env.step(action)
        state = next_state if next_state is not None else state

    final_balance = test_env.balance
    profit = final_balance - test_env.initial_balance
    print(f"\nTest results:")
    print(f"Final balance: ${final_balance:.2f}")
    print(f"Total profit: ${profit:.2f}")

if __name__ == "__main__":
    trained_agent, processed_data = train_agent()
    test_agent(trained_agent, processed_data)