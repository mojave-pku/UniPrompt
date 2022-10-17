import torch

def print_state_dict(state):
        for param_tensor in state:
            print(param_tensor,'\t',state[param_tensor].size())

def update_state_dict_for_two_tower(path, prompt_tower_init_type, prompt_tower_init_state_dict=None, low_tower_layers=8):
    assert prompt_tower_init_type in ['xlmr', 'roberta', 'random']
    with open(path, "rb") as f:
        xlm_roberta_state_dict = torch.load(f, map_location=torch.device("cpu"))

        if prompt_tower_init_type == 'roberta':
            assert prompt_tower_init_state_dict is not None
            with open(prompt_tower_init_state_dict, "rb") as rpt:
                r_state_dict = torch.load(rpt, map_location=torch.device("cpu"))

        new_two_tower_state_dict = {}

        def prefix_copy(src_prefix, dst_prefix, used_state_dict):
            src_keys = []
            for key in used_state_dict.keys():
                if key.startswith(src_prefix):
                    src_keys.append(key)
            
            for key in src_keys:
                new_two_tower_state_dict[dst_prefix+key[len(src_prefix):]] \
                    = used_state_dict[key]
        
        prefix_copy('roberta.embeddings.', 'roberta.embeddings.', xlm_roberta_state_dict)
        for i in range(low_tower_layers):
            if prompt_tower_init_type == 'xlmr':
                prefix_copy(f'roberta.encoder.layer.{i}.', f'prompt_tower.encoder.{i}.', xlm_roberta_state_dict)
            elif prompt_tower_init_type == 'roberta':
                prefix_copy(f'roberta.encoder.layer.{i}.', f'prompt_tower.encoder.{i}.', r_state_dict)
            elif prompt_tower_init_type == 'random':
                pass
            else:
                raise ValueError
            prefix_copy(f'roberta.encoder.layer.{i}.', f'context_tower.encoder.{i}.', xlm_roberta_state_dict)
        for j in range(12-low_tower_layers):
            old_layer = j + low_tower_layers
            prefix_copy(f'roberta.encoder.layer.{old_layer}.', f'high_tower.encoder.{j}.', xlm_roberta_state_dict)
        prefix_copy('roberta.pooler.', 'pooler.', xlm_roberta_state_dict)
        prefix_copy('lm_head.', 'lm_head.', xlm_roberta_state_dict)
    
    return new_two_tower_state_dict
