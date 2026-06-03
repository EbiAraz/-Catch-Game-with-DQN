import numpy as np
import random
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
import os, time
import platform
import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

# ═══════════════════════════════════════════
#  GPU/CUDA Configuration
# ═══════════════════════════════════════════
#  GPU/CUDA Configuration
# ═══════════════════════════════════════════

# Diagnosis GPU availability
print("🔍 GPU Detection...")
gpus = tf.config.list_physical_devices('GPU')
tf_has_cuda = tf.test.is_built_with_cuda()
if gpus:
    print(f"✅ Number of GPU found  : {len(gpus)}")
    for gpu in gpus:
        print(f"   - {gpu}")
    
    # set memory growth to prevent TensorFlow from allocating all GPU memory at once
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU Memory Growth  enabled")
    except RuntimeError as e:
        print(f"❌ Error setting GPU memory growth: {e}")
    
    # set precision policy for faster computation
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("✅ Mixed Precision (float16) enabled")
else:
    print("⚠️  No GPU found. Using CPU.")
    print(f"ℹ️ TensorFlow built with CUDA: {tf_has_cuda}")
    if platform.system() == "Windows" and not tf_has_cuda:
        print("ℹ️ Native Windows + TensorFlow >= 2.11 does not support CUDA GPU.")
        print("ℹ️ To use GPU: run TensorFlow inside WSL2 (Ubuntu) or use tensorflow-directml.")

# information of GPU
def print_gpu_info():
    print("\n" + "="*55)
    print("📊 GPU Information:")
    print("="*55)
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU Available: {len(gpus) > 0}")
    print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    print(f"TensorFlow Version: {tf.__version__}")
    print("="*55 + "\n")


# ═══════════════════════════════════════════
#  1. Gymnasium Environment
# ═══════════════════════════════════════════

class CatchEnv(gym.Env):
    """Custom Catch game environment compatible with Gymnasium"""
    
    def __init__(self, grid_size=10):
        super(CatchEnv, self).__init__()
        self.grid_size = grid_size
        
        # Definition of action and observation space
        self.action_space = spaces.Discrete(3)  # Left, Stay, Right
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32
        )  # [ball_x, ball_y, basket_x]
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = {
            'ball_x': random.randint(0, self.grid_size - 1),
            'ball_y': 0,
            'basket_x': self.grid_size // 2,
            'done': False
        }
        return self._get_state(), {}
    
    def _get_state(self):
        return np.array([
            self.game['ball_x'] / self.grid_size,
            self.game['ball_y'] / self.grid_size,
            self.game['basket_x'] / self.grid_size
        ], dtype=np.float32)
    
    def step(self, action):
        #  Basket Updating 
        if action == 0:
            self.game['basket_x'] = max(0, self.game['basket_x'] - 1)
        elif action == 2:
            self.game['basket_x'] = min(self.grid_size - 1, self.game['basket_x'] + 1)
        
        # Ball location updating
        self.game['ball_y'] += 1
        
        reward = 0.0
        terminated = False
        
        if self.game['ball_y'] >= self.grid_size - 1:
            terminated = True
            if abs(self.game['ball_x'] - self.game['basket_x']) <= 1:
                reward = 1.0
            else:
                reward = -1.0
        
        return self._get_state(), reward, terminated, False, {}
    
    def render(self):
        print("+" + "-" * self.grid_size + "+")
        for y in range(self.grid_size):
            row = ""
            for x in range(self.grid_size):
                if y == self.game['ball_y'] and x == self.game['ball_x']:
                    row += "O"
                elif y == self.grid_size - 1 and abs(x - self.game['basket_x']) <= 1:
                    row += "="
                else:
                    row += "."
            print("|" + row + "|")
        print("+" + "-" * self.grid_size + "+")
    
    def close(self):
        pass


GRID_SIZE = 10
STATE_SIZE = 3
ACTION_SIZE = 3
MAX_STEPS_PER_GAME = GRID_SIZE * 4
PLAY_STEP_DELAY = 0.22
TEST_TERMINAL_DELAY = 0.18
TEST_RENDER_FPS = 12


def apply_plot_theme(fig, axes):
    """Apply a consistent modern style for all training/test charts."""
    fig.patch.set_facecolor('#f6f7fb')
    for ax in axes:
        ax.set_facecolor('#ffffff')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#d6d8e0')
        ax.spines['bottom'].set_color('#d6d8e0')
        ax.grid(True, color='#dfe3ee', linewidth=0.9, alpha=0.9)


def _ask_int(prompt_text, default_value, min_value=None, max_value=None):
    raw = input(f"{prompt_text} [default: {default_value}]: ").strip()
    if raw == "":
        return default_value
    try:
        value = int(raw)
    except ValueError:
        print(f"⚠️ Invalid input. Using default: {default_value}")
        return default_value
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def _ask_float(prompt_text, default_value, min_value=None, max_value=None):
    raw = input(f"{prompt_text} [default: {default_value}]: ").strip()
    if raw == "":
        return default_value
    try:
        value = float(raw)
    except ValueError:
        print(f"⚠️ Invalid input. Using default: {default_value}")
        return default_value
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


def ask_runtime_visual_settings():
    """Return runtime visualization/test settings without interactive prompts."""
    return {
        "render_fps": TEST_RENDER_FPS,
        "render_games": 5,
        "test_delay": TEST_TERMINAL_DELAY,
        "play_delay": PLAY_STEP_DELAY,
        "max_steps": MAX_STEPS_PER_GAME,
    }

def reset_game():
    """Reset the game state"""
    return {
        'ball_x': random.randint(0, GRID_SIZE - 1),
        'ball_y': 0,
        'basket_x': GRID_SIZE // 2,
        'done': False
    }

def get_state(game):
    """Convert game state to normalized numpy array for neural network input"""
    return np.array([
        game['ball_x'] / GRID_SIZE,
        game['ball_y'] / GRID_SIZE,
        game['basket_x'] / GRID_SIZE
    ], dtype=np.float32)


# ═══════════════════════════════════════════
#  2. Build Neural Network with Keras (GPU-Optimized)
# ═══════════════════════════════════════════

def build_network():
    """GPU-optimized network architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(STATE_SIZE,)),
        tf.keras.layers.Dense(128, activation='relu', dtype='float32'),  # Input layer uses float32
        tf.keras.layers.Dense(128, activation='relu', dtype='float32'),
        tf.keras.layers.Dense(64, activation='relu', dtype='float32'),
        tf.keras.layers.Dense(ACTION_SIZE, activation='linear', dtype='float32')  # Output layer uses float32
    ])
    
    #  GPU-optimized
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, use_ema=True),  # EMA optimizer 
        loss='mse',
        metrics=['mae']
    )
    return model

def copy_weights(source, target):
    """Copy weights from source network to target network"""
    target.set_weights(source.get_weights())


# ═══════════════════════════════════════════
#  3. Experience Replay Memory
# ═══════════════════════════════════════════

def create_memory(capacity=10000):
    return deque(maxlen=capacity)

def push_memory(memory, state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def sample_memory(memory, batch_size):
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.int32),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(next_states, dtype=np.float32),
        np.asarray(dones, dtype=np.float32)
    )


# ═══════════════════════════════════════════
#  4. Action Selection
# ═══════════════════════════════════════════

def select_action(state, policy_net, epsilon):
    """ε-greedy"""
    if random.random() < epsilon:
        return random.randrange(ACTION_SIZE)

    state_input = tf.convert_to_tensor(state[None, :], dtype=tf.float32)  # (1, 3)
    q_values = policy_net(state_input, training=False)  # (1, 3)
    return int(tf.argmax(q_values[0]).numpy())


# ═══════════════════════════════════════════
#  5. Learning
# ═══════════════════════════════════════════

def learn(memory, policy_net, target_net, batch_size=64, gamma=0.99):
    """
    Q_target = r + γ × max Q_target(s', a')
    """
    if len(memory) < batch_size:
        return None

    states, actions, rewards, next_states, dones = \
        sample_memory(memory, batch_size)

    # ── Q target ──
    next_q = target_net(next_states, training=False).numpy()     # (64, 3)
    max_next_q = np.max(next_q, axis=1)                     # (64,)
    targets_value = rewards + (1 - dones) * gamma * max_next_q

    # ── Q current ──
    current_q = policy_net(states, training=False).numpy()        # (64, 3)

    # Vectorized update: assign updated Q-targets only for chosen actions.
    rows = np.arange(batch_size)
    current_q[rows, actions] = targets_value

    # train_on_batch avoids fit() overhead for single-step replay updates.
    history = policy_net.train_on_batch(states, current_q)
    if isinstance(history, (list, tuple)):
        return float(history[0])
    return float(history)


# ═══════════════════════════════════════════
#  5.5 Faster Learning with GradientTape
# ═══════════════════════════════════════════

# If you want faster learning, use this version:

optimizer_global = tf.keras.optimizers.Adam(learning_rate=0.001, use_ema=True)
loss_fn = tf.keras.losses.MeanSquaredError()

@tf.function  # GPU optimization: compile to graph
def _train_step(states_t, actions_t, rewards_t, next_states_t, dones_t, policy_net, target_net, batch_size, gamma):
    # Q target
    next_q = target_net(next_states_t, training=False)
    max_next_q = tf.reduce_max(next_q, axis=1)
    target_q = rewards_t + (1 - dones_t) * gamma * max_next_q

    with tf.GradientTape() as tape:
        # Q current
        all_q = policy_net(states_t, training=True)           # (64, 3)
        indices = tf.stack([
            tf.range(batch_size, dtype=tf.int32),
            actions_t
        ], axis=1)
        predicted_q = tf.gather_nd(all_q, indices)            # (64,)

        loss = loss_fn(target_q, predicted_q)

    # Optimization
    grads = tape.gradient(loss, policy_net.trainable_variables)
    # Gradient Clipping
    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    optimizer_global.apply_gradients(zip(grads, policy_net.trainable_variables))

    return loss

def learn_fast(memory, policy_net, target_net, batch_size=64, gamma=0.99):
    """Faster version with GradientTape - GPU Optimized"""
    if len(memory) < batch_size:
        return None

    states, actions, rewards, next_states, dones = \
        sample_memory(memory, batch_size)

    states_t = tf.convert_to_tensor(states, dtype=tf.float32)
    actions_t = tf.convert_to_tensor(actions, dtype=tf.int32)
    rewards_t = tf.convert_to_tensor(rewards, dtype=tf.float32)
    next_states_t = tf.convert_to_tensor(next_states, dtype=tf.float32)
    dones_t = tf.convert_to_tensor(dones, dtype=tf.float32)

    loss = _train_step(states_t, actions_t, rewards_t, next_states_t, dones_t, policy_net, target_net, batch_size, gamma)
    return float(loss.numpy())


# ═══════════════════════════════════════════
#  6. Training (GPU-Accelerated)
# ═══════════════════════════════════════════

def train(use_fast=True):
    print_gpu_info()
    print(f"🖥️  TensorFlow {tf.__version__}")
    
    # Display GPU and CPU devices
    print("📱 Available Devices:")
    for device in tf.config.list_logical_devices():
        print(f"   - {device}")
    print()
    
    gpus = tf.config.list_physical_devices('GPU')
    cpu_devices = tf.config.list_physical_devices('CPU')
    print(f"🔧 GPU devices: {len(gpus)}")
    print(f"🔧 CPU devices: {len(cpu_devices)}")

    # Build networks
    policy_net = build_network()
    target_net = build_network()
    copy_weights(policy_net, target_net)

    memory = create_memory(10000)

    # Create Gymnasium environment
    env = CatchEnv(grid_size=GRID_SIZE)

    # Hyperparameters
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.99
    batch_size = 64
    target_update = 10
    num_episodes = 2000

    all_scores = []
    all_losses = []
    avg_scores = []
    win_count = 0

    # Select learning function
    learn_func = learn_fast if use_fast else learn

    print("=" * 55)
    print("🎮  Starting DQN Training (TensorFlow) - Catch Game with Gymnasium")
    print(f"📝  Learning Method: {'GradientTape (Fast)' if use_fast else 'model.fit'}")
    if gpus:
        print(f"🚀 GPU Acceleration: Enabled ({len(gpus)} GPU)")
    else:
        print("💻 GPU Acceleration: Disabled (CPU Mode)")
    print("=" * 55)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = select_action(state, policy_net, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            push_memory(memory, state, action, reward, next_state, float(done))

            loss = learn_func(memory, policy_net, target_net, batch_size, gamma)
            if loss is not None:
                all_losses.append(loss)

            state = next_state
            total_reward += reward

        # After episode
        all_scores.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if total_reward > 0:
            win_count += 1

        # Update target network
        if episode % target_update == 0:
            copy_weights(policy_net, target_net)

        # Print progress
        if episode % 100 == 0:
            avg = np.mean(all_scores[-100:])
            avg_scores.append(avg)
            wr = win_count
            bar = "█" * (wr // 5) + "░" * (20 - wr // 5)

            print(f"📊 Episode {episode:5d} │ " f"Average: {avg:+.2f} │ " f"Wins: {wr:3d}% [{bar}] │ " f"ε: {epsilon:.3f}")
            win_count = 0
        
        # If epsilon reaches minimum, stop training and start playing
        if epsilon <= epsilon_min + 0.001:  # Small margin for precision
            print(f"\n✅ Epsilon reached minimum ({epsilon:.4f})! Stopping training...")
            break

    env.close()

    # Save model
    policy_net.save_weights("model_weights.weights.h5")
    print("\n💾 Model saved: model_weights.weights.h5")
    
    # Clear GPU memory
    if gpus:
        print("🧹 Clearing GPU memory...")
        tf.keras.backend.clear_session()

    plot_training(all_scores, avg_scores, all_losses)
    return policy_net, target_net, memory


# ═══════════════════════════════════════════
#  7. Plotting
# ═══════════════════════════════════════════

def plot_training(scores, avg_scores, losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    apply_plot_theme(fig, [ax1, ax2])

    ax1.plot(scores, alpha=0.35, color='#4f78ff', label='Episode reward')
    w = 50
    if len(scores) >= w:
        ma = np.convolve(scores, np.ones(w) / w, mode='valid')
        ax1.plot(range(w - 1, len(scores)), ma, color='#ff5a5f', linewidth=2.2, label=f'Moving Average {w}')
    ax1.set_title('Training Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()

    if losses:
        ax2.plot(losses, alpha=0.35, color='#8b5cf6', label='Step loss')
        if len(losses) >= 100:
            la = np.convolve(losses, np.ones(100) / 100, mode='valid')
            ax2.plot(range(99, len(losses)), la, color='#ef4444', linewidth=2.2, label='Loss MA(100)')
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('MSE')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    plt.show()
    print("📊 Saved: results.png")


def render_game_pygame(screen, clock, env, game_idx, num_games, fps=TEST_RENDER_FPS, cell_size=60):
    """Render current game state in a pygame window. Returns False if window is closed."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

    width = env.grid_size * cell_size
    height = env.grid_size * cell_size + 60

    bg = (20, 24, 36)
    grid = (60, 68, 90)
    ball_color = (255, 178, 46)
    basket_color = (99, 217, 187)
    text_color = (240, 245, 255)

    screen.fill(bg)

    for y in range(env.grid_size):
        for x in range(env.grid_size):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, grid, rect, 1)

    ball_x = env.game['ball_x']
    ball_y = env.game['ball_y']
    basket_x = env.game['basket_x']

    if 0 <= ball_y < env.grid_size:
        center = (
            ball_x * cell_size + cell_size // 2,
            ball_y * cell_size + cell_size // 2,
        )
        pygame.draw.circle(screen, ball_color, center, max(8, cell_size // 4))

    basket_y = env.grid_size - 1
    for bx in [basket_x - 1, basket_x, basket_x + 1]:
        if 0 <= bx < env.grid_size:
            rect = pygame.Rect(
                bx * cell_size + 4,
                basket_y * cell_size + cell_size // 2,
                cell_size - 8,
                max(10, cell_size // 3),
            )
            pygame.draw.rect(screen, basket_color, rect, border_radius=6)

    font = pygame.font.SysFont("Segoe UI", 24)
    status = font.render(f"Test Game {game_idx}/{num_games}", True, text_color)
    screen.blit(status, (12, env.grid_size * cell_size + 15))

    pygame.display.flip()
    clock.tick(max(1, int(fps)))
    return True


def test_and_plot(policy_net=None, num_games=30, render_games=5, render_delay=TEST_TERMINAL_DELAY, render_fps=TEST_RENDER_FPS, max_steps_per_game=MAX_STEPS_PER_GAME):
    """Run greedy test episodes, show graphical rendering, and save test plots."""
    if policy_net is None:
        policy_net = build_network()
        if not os.path.exists("model_weights.weights.h5"):
            print("❌ Error: Model weights file not found!")
            print("Please train the model first.")
            return None
        policy_net.load_weights("model_weights.weights.h5")

    env = CatchEnv(grid_size=GRID_SIZE)
    rewards = []
    wins = 0

    print("\n" + "=" * 60)
    print("🧪 Running test phase (greedy policy, epsilon=0)")
    print(f"🎬 Render games: {min(render_games, num_games)}")
    if max_steps_per_game is None:
        print("♾️ Step limiter: disabled")
    else:
        print(f"🛡️ Step limiter: {max_steps_per_game}")

    pygame_window_open = False
    screen = None
    clock = None
    if HAS_PYGAME and render_games > 0:
        try:
            pygame.init()
            cell_size = 60
            width = env.grid_size * cell_size
            height = env.grid_size * cell_size + 60
            screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Catch DQN Test Viewer")
            clock = pygame.time.Clock()
            pygame_window_open = True
            print("✅ Render backend: pygame")
        except Exception as e:
            pygame_window_open = False
            print(f"⚠️ Pygame window could not start: {e}")
            print("⚠️ Render backend: terminal")
    elif render_games > 0:
        print("⚠️ Pygame not installed. Falling back to terminal rendering.")
        print("   Install with: pip install pygame")
        print("⚠️ Render backend: terminal")
    else:
        print("ℹ️ Render backend: none (render_games=0)")

    print("=" * 60)

    for game_idx in range(1, num_games + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        should_render = game_idx <= render_games

        while not done:
            step_count += 1
            if max_steps_per_game is not None and step_count > max_steps_per_game:
                print(
                    f"⚠️ Step limit ({max_steps_per_game}) reached in test game {game_idx}; forcing episode end."
                )
                done = True
                break

            if should_render:
                if pygame_window_open:
                    try:
                        if not render_game_pygame(screen, clock, env, game_idx, num_games, fps=render_fps):
                            pygame_window_open = False
                            pygame.quit()
                            print("⚠️ Pygame window closed. Continuing without graphical render.")
                    except Exception as e:
                        pygame_window_open = False
                        pygame.quit()
                        print(f"⚠️ Pygame render failed: {e}")
                        print("⚠️ Continuing with terminal render.")
                else:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print(f"🧪 Test Game {game_idx}/{num_games}")
                    env.render()

            action = select_action(state, policy_net, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

            if should_render and not pygame_window_open:
                time.sleep(render_delay)

        rewards.append(total_reward)
        if total_reward > 0:
            wins += 1

        if should_render and not pygame_window_open:
            env.render()
            print("✅ Caught!" if total_reward > 0 else "❌ Missed!")
            time.sleep(0.5)

    env.close()
    if pygame_window_open:
        time.sleep(0.4)
        pygame.quit()

    rewards_np = np.array(rewards, dtype=np.float32)
    running_win_rate = np.cumsum(rewards_np > 0) / np.arange(1, len(rewards_np) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    apply_plot_theme(fig, [ax1, ax2])

    ax1.plot(rewards_np, color='#2962ff', alpha=0.8, label='Test reward')
    if len(rewards_np) >= 5:
        w = 5
        ma = np.convolve(rewards_np, np.ones(w) / w, mode='valid')
        ax1.plot(range(w - 1, len(rewards_np)), ma, color='#f43f5e', linewidth=2.2, label=f'MA({w})')
    ax1.set_title('Test Reward per Game')
    ax1.set_xlabel('Game')
    ax1.set_ylabel('Reward')
    ax1.legend()

    ax2.plot(running_win_rate * 100, color='#16a34a', linewidth=2.2)
    ax2.fill_between(range(len(running_win_rate)), running_win_rate * 100, color='#86efac', alpha=0.3)
    ax2.set_title('Running Win Rate (%)')
    ax2.set_xlabel('Game')
    ax2.set_ylabel('Win Rate %')
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150)
    plt.show()

    print("\n" + "=" * 60)
    print(f"✅ Test finished. Wins: {wins}/{num_games} ({wins / num_games * 100:.1f}%)")
    print(f"✅ Average test reward: {rewards_np.mean():+.3f}")
    print("📊 Saved: test_results.png")
    print("=" * 60)

    return {
        "wins": wins,
        "num_games": num_games,
        "win_rate": wins / num_games,
        "avg_reward": float(rewards_np.mean()),
    }


# ═══════════════════════════════════════════
#  8. Watching the game with learning
# ═══════════════════════════════════════════

def play(num_games=10, policy_net=None, target_net=None, memory=None, learn_func=None, step_delay=PLAY_STEP_DELAY, max_steps_per_game=MAX_STEPS_PER_GAME):
    """
    Watching the game with learning
    
    If policy_net, target_net, and memory are provided, 
    weights are updated during the game
    """
    
    # If learn_func is not provided, use learn_fast
    if learn_func is None:
        learn_func = learn_fast

    # If network is not provided, load it
    if policy_net is None:
        policy_net = build_network()
        
        if not os.path.exists("model_weights.weights.h5"):
            print("❌ Error: Model weights file not found!")
            print("Please train the model first (Option 1)")
            return
        
        policy_net.load_weights("model_weights.weights.h5")
        continue_learning = False
    else:
        continue_learning = True
        if target_net is None or memory is None:
            continue_learning = False

    # Create Gymnasium environment
    env = CatchEnv(grid_size=GRID_SIZE)

    wins = 0
    names = ["⬅️ Left", "⏸️ Stay", "➡️ Right"]

    print("\n" + "=" * 50)
    if continue_learning:
        print("🎮  Watching the game + continuous learning")
        print("⚙️  Weights and memory are being updated...")
    else:
        print("🎮  Watching the game (without learning)")
    if max_steps_per_game is None:
        print("♾️ Step limiter: disabled")
    else:
        print(f"🛡️ Step limiter: {max_steps_per_game}")
    print("=" * 50)

    # Learning hyperparameters during the game
    gamma = 0.99
    batch_size = 64
    epsilon_play = 0.05  # For some exploration
    
    target_update_play = 5
    game_count = 0

    for g in range(1, num_games + 1):
        state, _ = env.reset()
        done = False
        step_count = 0
        reward = -1.0

        while not done:
            step_count += 1
            if max_steps_per_game is not None and step_count > max_steps_per_game:
                print(
                    f"⚠️ Step limit ({max_steps_per_game}) reached in game {g}; forcing episode end."
                )
                done = True
                break

            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"🎮 Game {g}/{num_games}")
            env.render()

            # Select action (with some exploration)
            prev_state = state
            action = select_action(state, policy_net, epsilon_play if continue_learning else 0)
            print(f"Action: {names[action]}")

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # If learning, store in memory
            if continue_learning and done:
                push_memory(memory, prev_state, action, reward, next_state, float(done))
                
                # Learning
                loss = learn_func(memory, policy_net, target_net, batch_size, gamma)
                if loss is not None:
                    print(f"   📉 Loss: {loss:.4f}")

            state = next_state
            
            time.sleep(step_delay)

        env.render()
        if reward > 0:
            print("✅ Caught!")
            wins += 1
        else:
            print("❌ Missed!")
        
        time.sleep(1)
        
        # Update target network during the game
        game_count += 1
        if continue_learning and game_count % target_update_play == 0:
            copy_weights(policy_net, target_net)
            print("🔄 Target network updated")

    env.close()
    
    # Save updated weights
    if continue_learning:
        policy_net.save_weights("model_weights.weights.h5")
        print("\n💾 Updated weights saved: model_weights.weights.h5")
    
    print(f"\n📊 Result: {wins}/{num_games} ({wins/num_games*100:.0f}%)")


# ═══════════════════════════════════════════
#  9. Execution
# ═══════════════════════════════════════════

if __name__ == "__main__":
    # Display system information
    print_gpu_info()
    
    # Check if a trained model already exists
    weights_exist = os.path.exists("model_weights.weights.h5")
    
    print("=" * 50)
    if weights_exist:
        print("✅ Previous trained model found!")
        print("\n1) Watch the game (without learning)")
        print("2) Retrain from scratch")
        print("3) Watch + continuous learning")
        print("4) Test model (plot + graphical render)")
        choice = input("\nChoice: ").strip()
        
        if choice == "2":
            cfg = ask_runtime_visual_settings()
            print("\n🔄 Retraining...")
            policy_net, target_net, memory = train(use_fast=True)
            test_and_plot(
                policy_net=policy_net,
                num_games=30,
                render_games=cfg["render_games"],
                render_delay=cfg["test_delay"],
                render_fps=cfg["render_fps"],
                max_steps_per_game=cfg["max_steps"],
            )
            print("\n▶️  Starting game...")
            play(
                num_games=10,
                policy_net=policy_net,
                target_net=target_net,
                memory=memory,
                learn_func=learn_fast,
                step_delay=cfg["play_delay"],
                max_steps_per_game=cfg["max_steps"],
            )
        elif choice == "3":
            cfg = ask_runtime_visual_settings()
            print("\n📖 Loading model for continuous learning...")
            policy_net = build_network()
            target_net = build_network()
            policy_net.load_weights("model_weights.weights.h5")
            copy_weights(policy_net, target_net)
            memory = create_memory(10000)
            print("\n▶️  Starting game + continuous learning...")
            play(
                num_games=10,
                policy_net=policy_net,
                target_net=target_net,
                memory=memory,
                learn_func=learn_fast,
                step_delay=cfg["play_delay"],
                max_steps_per_game=cfg["max_steps"],
            )
            test_and_plot(
                policy_net=policy_net,
                num_games=30,
                render_games=cfg["render_games"],
                render_delay=cfg["test_delay"],
                render_fps=cfg["render_fps"],
                max_steps_per_game=cfg["max_steps"],
            )
        elif choice == "4":
            cfg = ask_runtime_visual_settings()
            test_and_plot(
                num_games=30,
                render_games=cfg["render_games"],
                render_delay=cfg["test_delay"],
                render_fps=cfg["render_fps"],
                max_steps_per_game=cfg["max_steps"],
            )
        else:
            cfg = ask_runtime_visual_settings()
            print("\n▶️  Starting game...")
            play(step_delay=cfg["play_delay"], max_steps_per_game=cfg["max_steps"])
            test_and_plot(
                num_games=30,
                render_games=cfg["render_games"],
                render_delay=cfg["test_delay"],
                render_fps=cfg["render_fps"],
                max_steps_per_game=cfg["max_steps"],
            )
    else:
        print("❌ No trained model found!")
        print("\n1) Train model")
        print("2) Exit")
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            cfg = ask_runtime_visual_settings()
            print("\n🎓 Starting training...")
            policy_net, target_net, memory = train(use_fast=True)
            test_and_plot(
                policy_net=policy_net,
                num_games=30,
                render_games=cfg["render_games"],
                render_delay=cfg["test_delay"],
                render_fps=cfg["render_fps"],
                max_steps_per_game=cfg["max_steps"],
            )
            print("\n▶️  Starting game...")
            play(
                num_games=10,
                policy_net=policy_net,
                target_net=target_net,
                memory=memory,
                learn_func=learn_fast,
                step_delay=cfg["play_delay"],
                max_steps_per_game=cfg["max_steps"],
            )
        else:
            print("Exiting...")
    print("=" * 50)