import json
import random
import os
from tqdm import tqdm

def sampling(index_file, info_file, scopes, nums, out_file):
    with open(index_file, 'r') as index, open(info_file, 'r') as info:
        indexs = json.load(index)
        infos = json.load(info)
    sampled_lists = []
    count = 0
    for item, info in zip(list(indexs.items()), infos):
        k, v = item[0], item[1]
        if info['examples'][0]['class'] != int(k):
            print('error!')
            return None
        for idx, scope in enumerate(scopes):
            if scope[0] <= info['total_count'] < scope[1]:
                sampled_list = random.sample(v, min(nums[idx], info['total_count']))
                sampled_lists.append(sampled_list)
                count += len(sampled_list)
                break
    print('count: ', count)
    with open(out_file, 'w') as o:
        json.dump(sampled_lists, o)
    return 

if __name__ == '__main__':
    index_file = 'results_all_final_index.jsonl'
    info_file = 'results_all_final_info.jsonl'
    scopes = [[0, 1000], [1000, 5000], [5000, 10000], [10000, 10000000]]
    nums = [4, 10, 20, 50]
    out_file = 'results_all_final_selected_index.jsonl'

    sampling(index_file, info_file, scopes, nums, out_file)