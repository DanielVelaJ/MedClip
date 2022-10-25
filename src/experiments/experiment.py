def single_experiment(config):
    pass
def ablation_experiment(configs):
    # Get experimental config
    keys, values = zip(*configs.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for config in permutation_dicts:
        print('running experiment')
        result=5
    
    
    pass