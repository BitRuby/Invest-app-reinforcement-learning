Overview
This script demonstrates a Deep Q-Network (DQN) agent interacting with a stock trading environment using Google stock data from January 1, 2012, to January 1, 2013. The goal is to train the agent to make profitable trading decisions (buy, sell, hold) based on historical stock prices.

Steps and Components
1. Download Stock Data: Use the yfinance library to download Google stock data. The data includes daily closing prices for the specified period.

2. Setup DQN Parameters: State Size: 4 (Money, Stock Price, Number of Stocks, Total Asset Value) Action Size: 3 (Buy, Sell, Hold) Starting Money: 5000 Episodes: 100 (Number of training iterations) Memory Length: 5000 (Capacity of the replay memory) Batch Size: 64 (Number of experiences sampled from memory for learning) Steps: Number of trading days in the dataset (data length - 1) Gamma: 0.95 (Discount factor for future rewards) Epsilon Decay: 0.995 (Decay rate for exploration) Epsilon Min: 0.01 (Minimum exploration rate) Epsilon: 1 (Initial exploration rate)

3. Model Loading/Saving: Load a pre-trained model if available, otherwise train a new one.

4. Environment Definition:

Env Class: Simulates the stock market environment.
State: Current day index, money, number of stocks owned, total asset value.
Actions: reset (initialize environment), step (execute a trading action: buy, sell, hold).
Rewards: -10 for invalid actions (buy without money, sell without stocks). 10 for profitable actions (increase in asset value). -10 for unprofitable actions (decrease in asset value). 0 for neutral actions (no change in asset value).
5. DQN Agent Definition: DQNAgent Class: Implements the Deep Q-Learning algorithm. Model Architecture: Input layer: 4 neurons (state size). Hidden layers: Two layers with 32 neurons each, ReLU activation. Output layer: 3 neurons (action size), linear activation. Functions: act: Chooses an action based on epsilon-greedy policy. remember: Stores experiences in memory. replay: Updates the model using experiences from memory to minimize the loss.

6. Training Process:

Initialization: Create the agent and environment, initialize the scaler.
Episodes: For each episode, reset the environment. For each step within an episode: Scale the current state. Agent selects an action. Execute the action in the environment to get new state, reward, and done flag. Store the experience in memory. Update the state. Accumulate rewards for performance tracking. Perform experience replay to train the model. Save the model periodically.
7. Execution: If a pre-trained model exists, load it and initialize the environment and scaler. Otherwise, create a new agent and train it over the specified number of episodes, saving the model after training.

Detailed Walkthrough
1. Downloading Stock Data The yfinance library downloads Google's stock data for the specified period, capturing the daily closing prices.

2. Environment and Agent Setup The environment (Env class) simulates trading with actions to buy, sell, or hold stocks. The agent (DQNAgent class) interacts with this environment, using a neural network to predict the best actions based on past experiences.

3. DQN Algorithm The agent learns by iterating through episodes, where each episode represents a trading period. At each step within an episode, the agent: Chooses an action based on the current state. Executes the action in the environment. Receives a reward and the new state. Stores this experience in memory. Periodically trains the neural network using a batch of random experiences from memory (experience replay).

4. Training During training, the agent continually refines its trading strategy by adjusting the neural network weights based on the rewards received from its actions. This process involves exploring different actions (exploration) and exploiting known profitable actions (exploitation) as the agent becomes more confident in its predictions. By the end of the training process, the agent should ideally learn to maximize its total asset value through informed trading decisions
