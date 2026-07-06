from stable_baselines3 import DQN
from environment.streaming_env import StreamingEnv


def main():
    env = StreamingEnv()
    model = DQN.load("dqn_streaming_model")

    obs, _ = env.reset()

    total_reward = 0

    for _ in range(100):

        action, _ = model.predict(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        print(
            f"bw={info['bandwidth']:.2f} | "
            f"br={info['bitrate']} | "
            f"buf={info['buffer']:.2f} | "
            f"r={reward:.3f}"
        )

        if terminated:
            break

    print("\nTotal Reward:", total_reward)


if __name__ == "__main__":
    main()