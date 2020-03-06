import numpy as np
from scipy import stats, optimize
import gym
from gym import spaces
from copy import deepcopy


class Item():
    def __init__(self, config):
        # self.env = env
        self.orig_config = config
        config = deepcopy(config)
        self.store = config['store']
        self.price = config['price']
        self.cost = config['cost']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.scale = config['scale']
        self.inventory = config['inventory']
        self.on_order = config['on_order']
        self.balance = config['balance']
        self.order_cost = config['order_cost']
        self.carrying_cost = config['carrying_cost']
        self.timestep = 0
        self.starting_balance = config['balance']
        self.current_demand = 0
        self.mean_demand = self.scale * (self.alpha / (self.alpha + self.beta))

    def calculate_demand(self):
        demand = int(np.random.beta(self.alpha, self.beta) * self.scale)
        return demand

    def sell(self):
        demand = self.current_demand
        sales = np.min((demand, self.inventory))
        self.inventory -= sales
        self.balance += sales * self.price

    def order(self, quantity):
        self.on_order[-1] = quantity
        self.balance -= (quantity * self.cost + self.order_cost)

    def receive_order(self):
        self.inventory += self.on_order.pop(0)
        self.on_order.append(0)

    def step(self, order_quantity):
        self.current_demand = self.calculate_demand()
        order_quantity = int(order_quantity)
        prev_balance = self.balance
        self.sell()
        self.receive_order()
        if order_quantity > 0:
            self.order(order_quantity)
        self.balance -= ((self.inventory + sum(self.on_order)) * self.cost * self.carrying_cost)
        self.timestep += 1
        return self.balance - prev_balance

    @property
    def observation(self):
        return dict(oh_oo=np.array([self.inventory] + self.on_order))

    def reset(self):
        self.__init__(self.orig_config)


class SeasonalItem(Item):
    def __init__(self, config):
        super(SeasonalItem, self).__init__(config=config)
        self.seasonal_amplitude = config['seasonal_amplitude']
        self.day_of_year = round(np.random.uniform(0, 365))

    def calculate_demand(self):
        seasonal_factor = self.seasonal_amplitude * (np.sin(self.day_of_year * np.pi * 2 / 365) / 2 + .5)
        demand = int(np.random.beta(self.alpha, self.beta) * self.scale * seasonal_factor)
        return demand

    @property
    def observation(self):
        return dict(oh_oo_doy=np.array([self.inventory] + self.on_order + [self.day_of_year/365]))


class DemandDependentLtItem(Item):
    def order(self, quantity):
        if self.current_demand > self.mean_demand:
            self.on_order[-3] += quantity
        else:
            self.on_order[-1] = quantity
        self.balance -= (quantity * self.cost + self.order_cost)


class OrderDependentLtItem(Item):
    def order(self, quantity):
        if quantity > 1:
            self.on_order[-4] += quantity
        else:
            self.on_order[-1] = quantity
        self.balance -= (quantity * self.cost + self.order_cost)

class LowOrderLowLtItem(Item):
    def order(self, quantity):
        if quantity <= 2:
            self.on_order[-4] += quantity
        else:
            self.on_order[-1] = quantity
        self.balance -= (quantity * self.cost + self.order_cost)

class ItemSimulatorGym(gym.Env):
    def __init__(self, item=None, item_config=None):
        if not ((item and not item_config) or (not item and item_config)):
            raise ValueError('Must provide item or item_config and not both')
        if item:
            self.item = item
        else:
            self.item = Item(item_config)
        try:
            action_max = self.item.scale * 2 * self.item.seasonal_amplitude # tied to specific way demand is created in item sim right now
        except AttributeError:
            action_max = self.item.scale * 2
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([action_max]))
        self.observation_space = spaces.Box(low=-np.finfo(np.float32).max, high=np.finfo(np.float32).max,
                                            shape=[len(self.observation)])
        self.num_envs = 1

    @property
    def observation(self):
        observation = [x for x in self.item.observation.values()][0]
        return observation

    def step(self, action):
        action = int(np.round(action))
        reward = self.item.step(action)
        observation = self.observation
        done = False
        if self.item.balance < 0 or self.item.timestep > 60:
            done = True
        if done:
            info = {'episode':
                        {'r': self.item.balance - self.item.starting_balance,
                         'l': self.item.timestep}}
        else:
            info = {}
        return (observation, reward, done, info)

    def reset(self):
        self.item.reset()
        return self.observation

class ItemSimulatorGym(gym.Env):
    def __init__(self, item=None, item_config=None,action_space_type='discrete'):
        if not ((item and not item_config) or (not item and item_config)):
            raise ValueError('Must provide item or item_config and not both')
        if item:
            self.item = item
        else:
            self.item = Item(item_config)
        try:
            action_max = self.item.scale * 2 * self.item.seasonal_amplitude # tied to specific way demand is created in item sim right now
        except AttributeError:
            action_max = self.item.scale * 2
        if action_space_type == 'discrete':
            self.action_space = spaces.Discrete(action_max)
        elif action_space_type == 'box':
            self.action_space = spaces.Box(low=np.array([0]), high=np.array([action_max]))
        else:
            raise ValueError('Must provide discrete or box action_space_type')
        self.observation_space = spaces.Box(low=-np.finfo(np.float32).max, high=np.finfo(np.float32).max,
                                            shape=[len(self.observation)])
        self.num_envs = 1

    @property
    def observation(self):
        observation = [x for x in self.item.observation.values()][0]
        return observation

    def step(self, action):
        action = int(np.round(action))
        reward = self.item.step(action)
        observation = self.observation
        done = False
        if self.item.balance < 0 or self.item.timestep > 60:
            done = True
        if done:
            info = {'episode':
                        {'r': self.item.balance - self.item.starting_balance,
                         'l': self.item.timestep}}
        else:
            info = {}
        return (observation, reward, done, info)

    def reset(self):
        self.item.reset()
        return self.observation



def run_bowersox_baseline(env, num_episodes):
    alpha = env.item.alpha
    beta = env.item.beta
    scale = env.item.scale
    demand_pdf = stats.beta(a=alpha, b=beta)
    unit_profit = env.item.price - env.item.cost
    mean_demand = scale * (alpha / (alpha + beta))
    demand_variance = demand_variance = scale ** 2 * (alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
    mean_lt = len(env.item.on_order)
    def calc_safety_stock(sl):
        z_sl = demand_pdf.ppf(sl)  # Using beta since there's no LT variance so convolution just results in another beta
        return z_sl * np.sqrt(mean_lt * demand_variance)

    if isinstance(env.item, OrderDependentLtItem):
        long_lt_probability = demand_pdf.cdf(1/scale)
        short_lt_probability = 1 - long_lt_probability
        short_lt = 2 # TODO remove hard coding of "2" for short lt and tie to actual item
        long_lt = len(env.item.on_order)
        mean_lt = mean_lt * long_lt_probability + short_lt * short_lt_probability
        lt_variance = long_lt_probability * (mean_lt - long_lt)**2 + short_lt_probability * (mean_lt - short_lt)**2

        def calc_safety_stock(sl):
            z_sl = stats.norm.ppf(sl)  # Using normal, but not clear what we should use given extreme non-normality
            return z_sl * np.sqrt(mean_lt * demand_variance + mean_demand**2 * lt_variance)

    if isinstance(env.item, LowOrderLowLtItem):
        short_lt_probability = demand_pdf.cdf(2 / scale)
        long_lt_probability = 1 - short_lt_probability
        short_lt = 2  # TODO remove hard coding of "2" for short lt and tie to actual item
        long_lt = len(env.item.on_order)
        mean_lt = mean_lt * long_lt_probability + short_lt * short_lt_probability
        lt_variance = long_lt_probability * (mean_lt - long_lt) ** 2 + short_lt_probability * (mean_lt - short_lt) ** 2

        def calc_safety_stock(sl):
            z_sl = stats.norm.ppf(sl)  # Using normal, but not clear what we should use given extreme non-normality
            return z_sl * np.sqrt(mean_lt * demand_variance + mean_demand ** 2 * lt_variance)

    # calculate optimal sl (from a purely transactional pov)
    marginal_cost = lambda sl: env.item.carrying_cost * env.item.cost * calc_safety_stock(
        sl) - unit_profit * demand_pdf.expect(lb=demand_pdf.ppf(sl))
    optimal_sl = optimize.fsolve(marginal_cost, x0=.7)
    safety_stock = calc_safety_stock(optimal_sl)
    order_up_to_level = mean_demand * (mean_lt + 1) + safety_stock
    episodes = 0
    obs = []
    rewards = []
    dones = []
    infos = []
    epinfos = []
    actions = []
    while episodes < num_episodes:
        action = order_up_to_level - sum(env.observation)
        ob, reward, done, info = env.step(action)
        maybeepinfo = info.get('episode')
        if maybeepinfo:
            epinfos.append(maybeepinfo)
        obs.append(ob)
        rewards.append(reward)
        dones.append(done)
        actions.append(action)
        if done:
            episodes += 1
            env.reset()

    return (actions, obs, rewards, dones, infos, epinfos)
