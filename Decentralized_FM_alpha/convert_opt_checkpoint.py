import torch
import json
import os
import argparse
import tqdm

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HF checkpoints')
    parser.add_argument('--model-name', type=str, default='meta-llama/Meta-Llama-3-8B', 
                        help='model-name')
    parser.add_argument('--save-path', type=str, default='./pretrained_models', 
                        help='model-name')
    args = parser.parse_args()
    
    config = AutoConfig.from_pretrained(args.model_name)
    config.save_pretrained(args.save_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(args.save_path)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    ## emb
    print('saving embs')
    item = {}
    item['embed_tokens.weight'] = model.state_dict()['model.embed_tokens.weight']
    item['rotary_emb.weight'] = model.state_dict()['model.rotary_emb.weight']
    torch.save(item, os.path.join(args.save_path, 'pytorch_embs.pt'))

    ## out
    print('saving lm_head')
    item = {}
    item['lm_head.weight'] = model.state_dict()['model.lm_head.weight']
    item['final_layer_norm.weight'] = model.state_dict()['model.decoder.final_layer_norm.weight']
    torch.save(item, os.path.join(args.save_path, 'pytorch_lm_head.pt'))
    
    print('saving layers')
    for i in tqdm.tqdm(range(0, config.num_hidden_layers)):
        layer_prefix = f'model.layers.{i}.'

        item = {}

        layer_maps = {k:v for k,v in model.state_dict().items() if k.startswith(layer_prefix)}

        for k, v in layer_maps.items():
            new_k = k.replace(layer_prefix, '')
            item[new_k] = v

        torch.save(item, os.path.join(args.save_path, f'pytorch_{i}.pt'))

        del item
    
