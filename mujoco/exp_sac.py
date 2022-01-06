import argparse

from trainers import *
from execution_plans import basic_experiment
import components.pytorch_util as ptu
ptu.set_gpu_mode(True)

args = argparse.Namespace(
    name='',
    trainer='',
    seed=None,

    device='cuda',
    exp_root='exp',
    checkpoint_freq=100_000,

    env=None,
    history_length=1,
    reward_clip=False,
    reward_std=0,
    discount=0.99,

    min_num_steps_before_training=1600,
    num_epochs=50,
    num_steps_per_epoch=10000,
    max_episode_length=int(108e3),
    num_eval_steps_per_epoch=int(5e5),
    eval_max_episode=20,

    capacity=int(5e5),
    multi_step=1,
    priority_weight=0.4,
    priority_exponent=0.5,

    lr=3e-4,
    batch_size=256,
    target_update_freq=1,
    tau=5e-3,
    network_size=[256, 256],
    grad_step_repeat=1,
    grad_step_period=1,

    dueling=True,
    epsilon_steps=200_000,
    lpuct=1,

    prioritized=False,
    vmin=-10,
    vmax=10,
    natoms=51,
    resampling_rate=None,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # required 1. general
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=str, required=True)
    parser.add_argument('--trainer', type=str, required=True)
    parser.add_argument('--env', type=str, required=True)

    # optional
    parser.add_argument('--tau', type=float, default=5e-3)
    parser.add_argument('--use_temperature_func', type=bool, default=True)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--target_entropy', default=None)
    parser.add_argument('--ensemble_size', default=5)
    parser.add_argument('--grad_step_repeat', type=int, default=1)
    parser.add_argument('--grad_step_period', type=int, default=1)

    args = vars(args)
    args.update(vars(parser.parse_args()))
    args = argparse.Namespace(**args)
    basic_experiment(args)
