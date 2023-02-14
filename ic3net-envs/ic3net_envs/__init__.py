from gym.envs.registration import register

register(
    id='PredatorPrey-v0',
    entry_point='ic3net_envs.predator_prey_env:PredatorPreyEnv',
)

register(
    id='TrafficJunction-v0',
    entry_point='ic3net_envs.traffic_junction_env:TrafficJunctionEnv',
)

register(
    id='CooperativeSearch-v0',
    entry_point='ic3net_envs.cooperative_search_env:CooperativeSearchEnv',
)

register(
    id='TreasureHunt-v0',
    entry_point='ic3net_envs.treasure_hunt_env:TreasureHuntEnv',
)

register(
    id='JointMonitoring-v0',
    entry_point='ic3net_envs.joint_monitoring_env:JointMonitoringEnv',
)