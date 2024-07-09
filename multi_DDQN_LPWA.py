from typing import List
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Flatten, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Add, Subtract
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from hashlib import sha256
import time
import json


# LoRaCommunication class
class LoRaCommunication:
    def __init__(self, distance, sensitivity):
        self.distance = distance
        self.sensitivity = sensitivity

    def calculate_snr(self, distance):
        snr = self.sensitivity - 20 * np.log10(distance / self.distance)
        return snr

    def is_communication_successful(self, distance):
        distance += 1e-8
        snr = self.calculate_snr(distance)
        if snr > 0:
            return True
        return False

class GridEnvironment:
    def __init__(self, grid_size, num_agents):
        self.grid_size = grid_size
        self.state_size = grid_size * grid_size
        self.num_agents = num_agents
        self.lora = LoRaCommunication(1, -120)
        self.observation_space = self.state_size
        self.action_space = 4  # 上、下、左、右の4つの行動
        self.reset()

    def reset(self):
        self.agent_positions = [np.random.randint(self.grid_size) for _ in range(self.num_agents)]
        self.goal_position = np.random.randint(self.grid_size)
        while self.goal_position in self.agent_positions:
            self.goal_position = np.random.randint(self.grid_size)
        state = self.encode_state()
        return state

    def encode_state(self):
        state = np.zeros((self.grid_size, self.grid_size))
        for idx, pos in enumerate(self.agent_positions):
            state[pos, pos] = idx + 1
        state[self.goal_position, self.goal_position] = self.num_agents + 1
        return state.flatten()

    def step(self, actions):
        rewards = []
        prev_positions = self.agent_positions.copy()

        for idx, action in enumerate(actions):
            self.move_agent(action, idx)
            self.agent_positions = np.clip(self.agent_positions, 0, self.grid_size - 1)
    
        distance = np.abs(self.agent_positions[0] - self.agent_positions[1])
        communication_successful = self.lora.is_communication_successful(distance)

        done = any(pos == self.goal_position for pos in self.agent_positions)

        for idx in range(self.num_agents):
            reward = 0

            # Reward for moving closer to the goal
            prev_goal_dist = np.abs(prev_positions[idx] - self.goal_position)
            curr_goal_dist = np.abs(self.agent_positions[idx] - self.goal_position)
            reward += (prev_goal_dist - curr_goal_dist)

            # Reward for colliding with other agents
            if self.agent_positions[idx] in np.concatenate((self.agent_positions[:idx], self.agent_positions[idx+1:])):
                reward -= 5

            # Reward for reaching the goal
            if done:
                reward += 10

            # Penalty for unsuccessful communication
            if not communication_successful:
                reward -= 2

            rewards.append(reward)

        state = self.encode_state()
        return state, rewards, done

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


class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.85, epsilon=1.0, learning_rate=0.05):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.95
        self.learning_rate = learning_rate
        self.regularization_strength = 0.02
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))  # Add an extra Dense layer
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # Check the shape of the state
        if state.shape != (1, self.state_size):
            state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
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


class DuelingDQNAgent(DQNAgent):
    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))  # Add an extra Dense layer

        # Add the dueling architecture
        model.add(Dense(self.action_size + 1, activation="linear"))
        model.add(Lambda(lambda x: Subtract()([x[:, :1], x[:, 1:]]) + x[:, 1:]))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model


class MultiAgentDuelingDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, gamma=0.85, epsilon=1.0, learning_rate=0.05):
        super().__init__(state_size, action_size, gamma, epsilon, learning_rate)
        self.target_model = self.build_model()

    def build_model(self):
        K.clear_session()
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))  # Add an extra Dense layer

        # Add the dueling architecture
        model.add(Dense(self.action_size + 1, activation="linear"))
        model.add(Lambda(lambda x: K.expand_dims(x[:, 0], -1) + x[:, 1:] - K.mean(x[:, 1:], axis=1, keepdims=True), output_shape=(action_size,)))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state.reshape(1, -1))[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class CentralizedCriticDQNAgent(MultiAgentDuelingDQNAgent):
    def __init__(self, state_size, action_size, gamma=0.85, epsilon=1.0, learning_rate=0.05):
        super().__init__(state_size, action_size, gamma, epsilon, learning_rate)
        self.shared_critic = self.build_model()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.shared_critic.predict(next_state.reshape(1, -1))[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
            # Update the shared critic as well
            self.shared_critic.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        super().update_target_model()
        self.shared_critic.set_weights(self.model.get_weights())


def train_agent(agents, env, episodes, batch_size, target_update_interval, early_stopping_patience, max_actions, total_rewards):
    num_agents = len(agents)
    state_size = env.state_size
    
    early_stopping_counter = 0

    for e in range(episodes):
        state = env.reset()
        state = state[np.newaxis, :]
        done = False
        step = 0
        episode_rewards = [0] * num_agents
        while not done:
            actions = [agent.act(state) for agent in agents]
            next_state, rewards, done = env.step(actions)
            next_state = next_state[np.newaxis, :]

            for i in range(num_agents):
                agents[i].remember(state, actions[i], rewards[i], next_state, done)
                episode_rewards[i] += rewards[i]  # 各エージェントの報酬を更新

            state = next_state
            step += 1
            if done:
                print(f"Episode {e + 1}: Max steps: {step}")
                break

        # max_actionsとtotal_rewardsの更新
        for i in range(num_agents):
            max_actions[i].append(step)
            total_rewards[i].append(episode_rewards[i])

        for agent in agents:
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if e % target_update_interval == 0:
            for agent in agents:
                agent.update_target_model()

        if early_stopping_patience:
            if e >= early_stopping_patience and all(max_actions[i][-1] >= max_actions[i][-early_stopping_patience] for i in range(num_agents)):
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0

            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    grid_size = 10
    num_agents = 3
    env = GridEnvironment(grid_size, num_agents)
    state_size = env.state_size
    action_size = 4
    episodes = 200
    batch_size = 64
    target_update_interval = 10

    early_stopping_patience = 12

    agents = [
        CentralizedCriticDQNAgent(state_size, action_size, gamma=0.9, epsilon=0.8, learning_rate=0.1),
        CentralizedCriticDQNAgent(state_size, action_size, gamma=0.85, epsilon=1.0, learning_rate=0.05),
        CentralizedCriticDQNAgent(state_size, action_size, gamma=0.8, epsilon=0.7, learning_rate=0.2)
    ]


    max_actions = {i: [] for i in range(num_agents)}
    total_rewards = {i: [] for i in range(num_agents)}

    # train_agentを適切に変更して、各エージェントのデータを個別に追跡する必要があるかも


    train_agent(
        agents, env, episodes, batch_size, target_update_interval, early_stopping_patience,
        max_actions, total_rewards
    )

    # エージェントごとに最大行動と報酬のプロットを作成
    for i, agent in enumerate(agents):
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(max_actions[i])
        plt.xlabel("Episode")
        plt.ylabel("Max Actions")
        plt.title(f"Agent {i + 1}: Max Actions")

        plt.subplot(122)
        plt.plot(total_rewards[i])
        plt.xlabel("Episode")
        plt.ylabel("Total Rewards")
        plt.title(f"Agent {i + 1}: Total Rewards")
        plt.show()

    # テストフェーズ
    test_episodes = 10
    for e in range(test_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        while not done:
            actions = [agent.act(state) for agent in agents]
            next_state, rewards, done = env.step(actions)
            state = np.reshape(next_state, [1, state_size])
            if done:
                print(f"Test Episode {e+1} done.")
                break
