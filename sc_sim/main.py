#from sc_sim.simulation import Item, ItemSimulatorGym, run_bowersox_baseline
from sc_sim import simulation

from baselines.ppo2 import ppo2
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines import logger
import gym
from gym import spaces
import numpy as np

def build_env(): #no args because of the way DummyVecEnv works
    item = simulation.LowOrderLowLtItem(item_config)
    env = simulation.ItemSimulatorGym(item,action_space_type='discrete')
    return env

def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)

if __name__ =='__main__':
    num_timesteps = 2000000

    inventory_cost = .05
    balance = 200
    order_cost = 0
    on_order = [0, 0, 0, 0, 0]
    item_config = dict(store=None, price=10, cost=5, alpha=2, beta=5, scale=10,
                       inventory=0, on_order=on_order, carrying_cost=inventory_cost, balance=balance,
                       order_cost=order_cost)
    #log_path = 'logs/sc_sim'
    #configure_logger(log_path)

    # configure_logger('logs/mlp_16')
    # env = DummyVecEnv([build_env])
    # model = ppo2.learn(
    #     network='mlp',
    #     env=env,
    #     total_timesteps=num_timesteps,
    #     load_path= 'models/mlp_16',
    #     num_hidden = 16,
    #     num_layers = 2
    #     )
    # model.save('models/mlp_16')

    # configure_logger('logs/mlp_64_regular_item')
    # env = DummyVecEnv([build_env])
    # model = ppo2.learn(
    #     network='mlp',
    #     env=env,
    #     total_timesteps=num_timesteps,
    #     #load_path= 'model/mlp_64_low_order_item',
    #     save_interval = 500000
    #     )
    # model.save('model/mlp_64_regular_item')

    env = build_env()
    actions, obs, rewards, dones, infos, epinfos = simulation.run_bowersox_baseline(env, 10000)
    avg_reward = sum([x['r'] for x in epinfos])/len(epinfos)
    print(np.mean(avg_reward))