from data.data_loader import download_data, preprocess_data
from environment.trading_env import TradingEnvironment
from model.agent import DQNAgent
from config import config
import torch
import os, argparse
import random
import numpy as np

def train_agent():
    raw_data = download_data(config['symbol'], config['start_date'], config['end_date'])
    data = preprocess_data(raw_data)
    
    env = TradingEnvironment(data, config['initial_balance'])
    agent = DQNAgent(config)
    
    for episode in range(config['training']['episodes']):
        state = env.reset()
        total_reward = 0

        while state is not None:
            action = agent.act(state, config['actions'])
            next_state, reward = env.step(action)
            agent.remember(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        agent.replay(config['training']['batch_size'])
        print(f'Episode {episode+1}/{config['training']['episodes']}, total reward: {total_reward}')

    print('Training complete!')

    os.makedirs(os.path.dirname(config['model']['model_path']), exist_ok=True)
    torch.save(agent.model.state_dict(), config['model']['model_path'])
    print(f'Model saved to {config['model']['model_path']}')

    return data, agent

def backtest_agent(agent, data='None'):
    if data is None:
        raw_data = download_data(config['symbol'], config['start_date'], config['end_date'])
        data = preprocess_data(raw_data)

    test_env = TradingEnvironment(data, config['initial_balance'])
    state = test_env.reset()

    while state is not None:
        action = agent.act(state, config['actions'], is_backtesting=True)
        next_state, _ = test_env.step(action)
        state = next_state

    final_balance = test_env.balance
    profit = final_balance - test_env.initial_balance
    print(f'\nTest results:')
    print(f'- Final balance: ${final_balance:.2f}')
    print(f'- Total profit: ${profit:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DQN Stock Trading'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'backtest'],
        help='Choose "train" to train the model or "backtest" to only evaluate',
        default='train'
    )

    if parser.parse_args().mode == 'train':
        processed_data, trained_agent  = train_agent()
        backtest_agent(trained_agent, processed_data)
    else:
        trained_agent = DQNAgent(config)
        trained_agent.model.load_state_dict(torch.load(config['model']['model_path']))
        trained_agent.model.eval()
        backtest_agent(trained_agent, None)