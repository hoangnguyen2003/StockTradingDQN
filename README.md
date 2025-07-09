# StockTradingDQN

### Table of contents
* [Introduction](#star2-introduction)
* [Installation](#wrench-installation)
* [How to run](#zap-how-to-run) 
* [Contact](#raising_hand-questions)

## :star2: Introduction

* <p align="justify">Developed a reinforcement learning system for automated stock trading using Deep Q-Networks.</p>
* <p align="justify">Data pipeline for fetching and preprocessing stock market data (Yahoo Finance).</p>
* <p align="justify">Custom trading environment with buy/sell/hold actions.</p>
* <p align="justify">The AI agent learns optimal trading strategies (buy/hold/sell) by analyzing historical price data, moving averages, and returns.</p>
* <p align="justify">Achieved profitable trading decisions through Q-learning with experience replay and epsilon-greedy exploration.</p>

## :wrench: Installation

<p align="justify">Step-by-step instructions to get you running StockTradingDQN:</p>

### 1) Clone this repository to your local machine:

```bash
git clone https://github.com/hoangnguyen2003/StockTradingDQN.git
```

A folder called `StockTradingDQN` should appear.

### 2) Install the required packages:

Make sure that you have Anaconda installed. If not - follow this [miniconda installation](https://www.anaconda.com/docs/getting-started/miniconda/install).

<p align="justify">You can re-create our conda enviroment from `environment.yml` file:</p>

```bash
cd StockTradingDQN
conda env create --file environment.yml
```

<p align="justify">Your conda should start downloading and extracting packages.</p>

### 3) Activate the environment:

Your environment should be called `StockTradingDQN`, and you can activate it now to run the scripts:

```bash
conda activate StockTradingDQN
```

## :zap: How to run 
<p align="justify">To train StockTradingDQN:</p>

```bash
python main.py
```

To backtest the trained model, you can change the `symbol` to test on different stocks and adjust the `start_date/end_date` for different time periods in `config/config.yaml`.

```bash
python main.py --mode backtest
```

## :raising_hand: Questions
If you have any questions about the code, please contact Hoang Van Nguyen (hoangvnguyen2003@gmail.com) or open an issue.