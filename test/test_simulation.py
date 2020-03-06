import pytest
import sys
print(sys.path)

from sc_sim.simulation import Item, ItemSimulatorGym, run_bowersox_baseline

inventory_cost = .0004109589
balance = 1000
order_cost = 0
on_order = [0, 0, 0, 0, 0]
item_config = dict(store=None, price=10, cost=5, alpha=2, beta=5, scale=10,
                   inventory=0, on_order=on_order, carrying_cost=inventory_cost, balance=balance, order_cost=order_cost)

def test_Item_init():
    item = Item(item_config)
    assert item

def test_ItemSimulatorGym_init():
    env = ItemSimulatorGym(item_config=item_config)
    assert env

def test_ItemSimulatorGym_init_item():
    item = Item(item_config)
    env = ItemSimulatorGym(item=item)
    assert env

def test_ItemSimulatorGym_input_error():
    item = Item(item_config)
    with pytest.raises(ValueError):
        ItemSimulatorGym(item=item, item_config=item_config)

def test_ItemSimulatorGym_step():
    env = ItemSimulatorGym(item_config=item_config)
    reward = env.step(1)
    expected_balance = item_config['balance'] - item_config['cost'] + item_config['order_cost']
    assert env.item.balance == expected_balance
    print(item_config['on_order'])
    expected_oo = item_config['on_order'].copy()
    expected_oo.pop(0)
    expected_oo.append(1)
    assert env.item.on_order == expected_oo

def test_ItemSimulatorGym_reset():
    env = ItemSimulatorGym(item_config=item_config)
    env.step(1)
    env.reset()
    for key in item_config.keys():
        assert item_config[key] == getattr(env.item, key)

def test_bowersox_baseline():
    env = ItemSimulatorGym(item_config=item_config)
    run_bowersox_baseline(env, 100, .95)