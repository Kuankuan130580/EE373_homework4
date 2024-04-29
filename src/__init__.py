from gymnasium.envs.registration import register
from slot_machine import SlotMachines

register(
    id='{}-{}'.format('SlotMachines', 'v0'),
    entry_point=SlotMachines,
    max_episode_steps=1,
    nondeterministic=True)


