from stable_baselines3 import DQN
from environment.streaming_env import StreamingEnv


def main():
    env = StreamingEnv()

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        tensorboard_log="logs",
    )

    model.learn(total_timesteps=50000)

    model.save("dqn_streaming_model")

    print("Training finished.")


if __name__ == "__main__":
    main()