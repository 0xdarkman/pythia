import argparse

from pythia.trainer.shares_reinforcement import run_shares_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-dir",
        required=True
    )
    parser.add_argument(
        "--holding-tokens",
        default=None
    )
    parser.add_argument(
        "--buying-tokens",
        required=True
    )
    parser.add_argument(
        "--balance",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--epsilon-episode-start",
        type=int,
        default=1
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000
    )

    args = parser.parse_args()

    run_shares_model(
        holding_tokens=args.holding_tokens,
        buying_tokens=args.buying_tokens,
        starting_balance=args.balance,
        window=args.window,
        hidden_layers=[100],
        learning_rate=args.learning_rate,
        memory_size=args.memory_size,
        epsilon_episode_start=args.epsilon_episode_start,
        gamma=args.gamma,
        alpha=args.alpha,
        num_steps=args.num_steps,
        episodes=args.episodes,
        output_dir=args.job_dir
    )
