import numpy as np
import random
from collections import deque
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Add, Subtract, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from hashlib import sha256
import time
import json

class OffChain:
    def __init__(self):
        self.transactions = []

    def add_transaction(self, transaction):
        self.transactions.append(transaction)

    def clear_transactions(self):
        self.transactions = []

    def get_transactions(self):
        return self.transactions

# LoRaCommunication class
class LoRaCommunication:
    def __init__(self, distance, sensitivity):
        self.distance = distance
        self.sensitivity = sensitivity

    def calculate_snr(self, distance):
        if distance == 0:
            return float('inf')  # あるいは、適切な SNR 値を返す
        snr = self.sensitivity - 20 * np.log10(distance / self.distance)
        return snr

    def is_communication_successful(self, distance):
        snr = self.calculate_snr(distance)
        if snr > 0:
            return True
        return False

class Block:
    def __init__(self, index, transactions, timestamp, previous_hash):
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({"index": int(self.index), "transactions": [{k: v for k, v in tx.items()} for tx in self.transactions], "timestamp": int(self.timestamp), "previous_hash": self.previous_hash.encode().hex(), "nonce": int(self.nonce)}, sort_keys=True, default=int).encode()
        return sha256(block_string).hexdigest()

    def mine_block(self, difficulty):
        while self.hash[:difficulty] != "0" * difficulty:
            self.nonce += 1
            self.hash = self.calculate_hash()

class Blockchain:
    def __init__(self, difficulty=1):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        self.difficulty = difficulty

    def create_genesis_block(self):
        return Block(0, [], time.time(), "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def mine_block(self):
        new_block = Block(len(self.chain), self.pending_transactions, time.time(), self.get_latest_block().hash)
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        self.pending_transactions = []

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                return False
            if current_block.previous_hash != previous_block.hash:
                return False
        return True

class GridEnvironment:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.state_size = grid_size * grid_size
        self.lora = LoRaCommunication(1, -120)  # distance = 1, sensitivity = -120
        self.reset()

    def reset(self):
        self.agent_positions = [np.random.randint(self.grid_size) for _ in range(2)]
        self.goal_position = np.random.randint(self.grid_size)
        while self.goal_position in self.agent_positions:
            self.goal_position = np.random.randint(self.grid_size)
        state = self.encode_state()
        return state

    def encode_state(self):
        state = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        state[self.agent_positions[0], self.agent_positions[1]] = 1
        state[self.goal_position, self.goal_position] = 2
        return state.reshape(-1)  # Convert the NumPy array to a Python list

    def step(self, action1, action2):
        self.move_agent(action1, 0)
        self.move_agent(action2, 1)
        self.agent_positions = np.clip(self.agent_positions, 0, self.grid_size - 1)
        distance = np.abs(self.agent_positions[0] - self.agent_positions[1])
        communication_successful = self.lora.is_communication_successful(distance)

        done = (self.agent_positions[0] == self.goal_position) or (self.agent_positions[1] == self.goal_position)
        reward = -1
        if done:
            reward = 10

        if not communication_successful:
            reward -= 2

        state = self.encode_state()
        return state, reward, done


    def move_agent(self, action, agent_idx):
        if action == 0:
            if self.agent_positions[agent_idx] - 1 >= 0:
                self.agent_positions[agent_idx] -= 1
        elif action == 1:
            if self.agent_positions[agent_idx] + 1 < self.grid_size:
                self.agent_positions[agent_idx] += 1
        elif action == 2:
            if self.agent_positions[agent_idx] - self.grid_size >= 0:
                self.agent_positions[agent_idx] -= self.grid_size
        elif action == 3:
            if self.agent_positions[agent_idx] + self.grid_size < self.state_size:
                self.agent_positions[agent_idx] += self.grid_size
        else:
            raise ValueError("Invalid action")

class DuelingDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.65
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.95
        self.learning_rate = 0.015
        self.regularization_strength = 0.02
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=(self.state_size,))
        x = Dense(24, activation="relu", kernel_regularizer=l2(self.regularization_strength))(input_layer)
        x = Dense(24, activation="relu", kernel_regularizer=l2(self.regularization_strength))(x)

        value = Dense(1, activation="linear", kernel_regularizer=l2(self.regularization_strength))(x)
        advantages = Dense(self.action_size, activation="linear", kernel_regularizer=l2(self.regularization_strength))(x)
        q_values = Lambda(lambda x: x[0] + x[1] - K.mean(x[1], axis=1, keepdims=True))([value, advantages])

        model = Model(inputs=input_layer, outputs=q_values)
        model.compile(loss="mse", optimizer=RMSprop(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.expand_dims(state, axis=0))
        return np.argmax(act_values[0])

    def replay(self, batch_size, blockchain):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.expand_dims(next_state, axis=0))[0])
            target_f = self.model.predict(np.expand_dims(state, axis=0))
            target_f[0][action] = target
            self.model.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class MultiAgentDuelingDQNAgent(DuelingDQNAgent):
    def replay(self, batch_size, blockchain):
        batch_size = min(batch_size, len(blockchain.chain))
        minibatch = random.sample(blockchain.chain, batch_size)
        for block in minibatch:
            transactions = block.transactions
            for transaction in transactions:
                state = np.array(transaction["state"])
                action = transaction["actions"][0]  # Only agent1's action is used in this example
                reward = transaction["rewards"][0]  # Only agent1's reward is used in this example
                next_state = np.array(transaction["next_state"])
                done = transaction["done"]

                target = reward
                if not done:
                    target = reward + self.gamma * np.amax(self.model.predict(np.expand_dims(next_state, axis=0))[0])
                target_f = self.model.predict(np.expand_dims(state, axis=0))
                target_f[0][action] = target
                self.model.fit(np.expand_dims(state, axis=0), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(agent1, agent2, env, blockchain, offchain, episodes, batch_size, early_stopping_rounds):
    max_actions1 = []
    max_actions2 = []
    total_rewards1 = []
    total_rewards2 = []
    combined_rewards = []

    best_reward = float("-inf")
    no_improvement_rounds = 0
    early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_rounds)

    for e in range(episodes):
        state = env.reset()
        total_reward1 = 0
        total_reward2 = 0
        total_reward = 0

        for time in range(500):
            action1 = agent1.act(state)
            action2 = agent2.act(state)

            next_state, reward, done = env.step(action1, action2)
            agent1.remember(state, action1, reward, next_state, done)
            agent2.remember(state, action2, reward, next_state, done)

            state = next_state
            total_reward1 += reward
            total_reward2 += reward
            total_reward += reward

            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent1.epsilon:.2}")
                break

            if len(agent1.memory) > batch_size:
                agent1.replay(batch_size, blockchain)
                agent2.replay(batch_size, blockchain)

            transaction = {"state": state.tolist(), "actions": [action1, action2], "rewards": [reward, reward], "next_state": next_state.tolist(), "done": str(done)}

            offchain.add_transaction(transaction)

            if len(offchain.transactions) >= 10:
                process_offchain_transactions(blockchain, offchain)

        max_actions1.append(time)
        max_actions2.append(time)
        total_rewards1.append(total_reward1)
        total_rewards2.append(total_reward2)
        combined_rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            no_improvement_rounds = 0
        else:
            no_improvement_rounds += 1

        if no_improvement_rounds >= early_stopping_rounds:
            print("Early stopping triggered.")
            break

    return max_actions1, max_actions2, total_rewards1, total_rewards2, combined_rewards

def process_offchain_transactions(blockchain, offchain):
    if len(offchain.get_transactions()) > 0:
        for transaction in offchain.get_transactions():
            blockchain.add_transaction(transaction)
        offchain.clear_transactions()
        blockchain.mine_block()

def main():
    grid_size = 4
    env = GridEnvironment(grid_size)
    state_size = grid_size * grid_size
    action_size = 4

    agent1 = MultiAgentDuelingDQNAgent(state_size, action_size)
    agent2 = MultiAgentDuelingDQNAgent(state_size, action_size)
    blockchain = Blockchain()

    episodes = 300
    batch_size = 16
    early_stopping_rounds = 12

    offchain = OffChain()

    max_actions1, max_actions2, total_rewards1, total_rewards2, combined_rewards = train_agent(
        agent1, agent2, env, blockchain, offchain, episodes, batch_size, early_stopping_rounds)

    plt.plot(max_actions1, label="Agent 1")
    plt.plot(max_actions2, label="Agent 2")
    plt.xlabel("Episodes")
    plt.ylabel("Max Actions")
    plt.legend()
    plt.show()

    plt.plot(total_rewards1, label="Agent 1")
    plt.plot(total_rewards2, label="Agent 2")
    plt.plot(combined_rewards, label="Combined")
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
