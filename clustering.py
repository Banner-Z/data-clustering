import json
# import torch
import numpy as np
# from sklearn.cluster import KMeans
import argparse
from pathlib import Path
import pickle
import os
# from numba import config
# config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = True
import cuml
from angle_emb import Prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm
# import faiss
# from faiss import normalize_L2

def clustering(args):

    json_data = []
    with open(args.vec_path, 'r') as f:
        for l in f.readlines():
            try:
                json_data.append(json.loads(l))
            except:
                # print(l)
                continue
    x_list = [item[0] for item in json_data[:args.training_samples]]
    x = np.array(x_list, dtype=np.float32)

    ## faiss
    # ncentroids = 10
    # niter = 100
    # verbose = True
    # d = x.shape[1]
    # print(d)
    # kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True, seed=42, nredo=1, min_points_per_centroid = 3000, max_points_per_centroid=100000, update_index=True)
    # kmeans.train(x)
    # print(kmeans.centroids)
    # return x, kmeans

    ## sklearn
    # kmeans = KMeans(n_clusters=10)
    # kmeans.fit(x)
    # labels = kmeans.labels_
    # print(kmeans.cluster_centers_)

    ## cuml
    if args.clusterer == 'dbscan':
        if os.path.exists(os.path.join(args.output_dir, 'dbscan.pkl')):
            with open(os.path.join(args.output_dir, 'dbscan.pkl'), 'rb') as f:
                clusterer = pickle.load(f)
        else:
            clusterer = cuml.DBSCAN(eps = args.eps, min_samples = args.min_samples, metric='cosine', calc_core_sample_indices=True)
            y_pred = clusterer.fit_predict(x, out_dtype='int64')
            with open(os.path.join(args.output_dir, 'dbscan.pkl'), 'wb+') as f:
                _ = pickle.dump(clusterer, f)
    else:
        if os.path.exists(os.path.join(args.output_dir, 'kmeans.pkl')):
            with open(os.path.join(args.output_dir, 'kmeans.pkl'), 'rb') as f:
                clusterer = pickle.load(f)
        else:
            clusterer  = cuml.KMeans(n_clusters=args.num_clusters, verbose=True, \
                                        max_iter=args.max_iter)
            y_pred = clusterer.fit_predict(x)
            with open(os.path.join(args.output_dir, 'kmeans.pkl'), 'wb+') as f:
                _ = pickle.dump(clusterer, f)
    # print(kmeans.cluster_centers_)
    return x, clusterer

def matching_and_save(xb, xq, args):
    import faiss
    from faiss import normalize_L2

    normalize_L2(xb)
    normalize_L2(xq)
    d=xb.shape[1]

    index = faiss.IndexFlatIP(d)
    # index = faiss.IndexFlatL2(d)
    index.add(xb)
    D, I = index.search(xq, args.n_knn)
    print(I)
    with open(os.path.join(args.output_dir, 'knn_results.json'), 'w') as o, open(args.query_path, 'r') as f:
        data = []
        for l in f.readlines():
            data.append(json.loads(l))
        records = []
        for i, item in enumerate(I):
            knn_list = []
            for j in item:
                knn_list.append({'query': data[j]['query'], 'source': data[j]['source']})
            record = {
                'id': i,
                'knn': knn_list,
            }
            records.append(record)
        json.dump(records, o, indent=4)

def predict_only(args):

    # 加载embedding
    if not args.has_vec:
        peft_model_id = "angle-llama-7b-nli-v2"
        config = PeftConfig.from_pretrained(peft_model_id)
        tokenizer = AutoTokenizer.from_pretrained("Llama-2-7b-hf")
        model = (
            AutoModelForCausalLM.from_pretrained("Llama-2-7b-hf")
            .bfloat16()
            .cuda()
        )
        model = PeftModel.from_pretrained(model, peft_model_id).cuda()
    else:
        with open(args.vec_path, 'r') as f:
            vec_data = []
            for l in f:
                vec_data.append(json.loads(l))

    def decorate_text(text: str):
        return Prompts.A.format(text=text)
        
    # 加载聚类模型
    if args.clusterer == 'dbscan':
        with open(os.path.join(args.output_dir, 'dbscan.pkl'), 'rb') as f:
            clusterer = pickle.load(f)
    else:
        with open(os.path.join(args.output_dir, 'kmeans.pkl'), 'rb') as f:
            clusterer = pickle.load(f)

    # 预测
    with open(args.source_path, "r") as fr, open(args.labeled_file_path, 'a+') as fo:
        for idx, l in enumerate(tqdm(fr)):
            if args.start <= idx < args.end:
                d = json.loads(l)
                inputs = d['query']

                # 生成embedding
                if not args.has_vec:
                    tok = tokenizer([decorate_text(inputs)], return_tensors="pt")
                    if len(tok['input_ids'][0]) > 4094:
                        print("Too long: {}".format(len(tok['input_ids'][0])))
                        continue
                    # import pdb;pdb.set_trace()
                    for k, v in tok.items():
                        tok[k] = v.cuda()
                    vec = (
                        model(output_hidden_states=True, **tok)
                        .hidden_states[-1][:, -1]
                        .float()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                else:
                    vec = np.array(vec_data[idx]).astype(np.float32)

                # 预测结果并保存结果
                cluster = clusterer.predict(vec)[0]
                d['class'] = int(cluster)
                fo.write(json.dumps(d, ensure_ascii=False) + '\n')

def test(xb, xq, args):

    import faiss
    from faiss import normalize_L2
    d=xb.shape[1]

    # index = faiss.IndexFlatIP(d)
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    D, I = index.search(np.array([xb[15]]), args.n_knn)
    print(I)
    with open(os.path.join(args.output_dir, 'test.json'), 'w') as o, open(args.query_path, 'r') as f:
        data = []
        for i, l in enumerate(f.readlines()):
            data.append(json.loads(l))
        records = []
        for i, item in enumerate(I):
            knn_list = []
            for j in item:
                knn_list.append({'query': data[j]['query'], 'source': data[j]['source']})
            record = {
                'id': i,
                'knn': knn_list,
            }
            records.append(record)
        json.dump(records, o, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--query-path', required=True, type=str)
    parser.add_argument('--num-clusters', required=True, type=int)
    parser.add_argument('--training-samples', default=500000, type=int)
    parser.add_argument('--output-dir', required=True, type=Path, help="the clusterer model path")
    parser.add_argument('--vec-path', default=None, type=str, help="the vectors path")
    parser.add_argument('--source-path', default=None, type=str)
    parser.add_argument('--labeled-file-path', default=None, type=str, help="the clustering result path")
    # parser.add_argument('--do_svd', default=False, type=bool)
    # parser.add_argument('--init_type', default='scalable-k-means++', type=str)
    parser.add_argument('--max-iter', default=300, type=int)
    parser.add_argument('--clusterer', default='kmeans', type=str)
    parser.add_argument('--n-knn', default=50, type=int)
    parser.add_argument('--eps', default=0.5, type=float, help="the hyperparameter of DBSCAN")
    parser.add_argument('--min-samples', default=5, type=int, help="the hyperparameter of DBSCAN")
    parser.add_argument('--predict-only', action='store_true')
    parser.add_argument('--has-vec', action='store_true')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=1000000000, type=int)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.predict_only:
        print("start training clusterer...")
        xb, clusterer = clustering(args)
        if args.clusterer == 'dbscan':
            print('nums: {}'.format(len(clusterer.core_sample_indices_)))
            xq = xb[clusterer.core_sample_indices_]
        else:
            xq = clusterer.cluster_centers_
        print("start matching...")
        matching_and_save(xb, xq, args)
        # test(xb, xq, args)
    else:
        print("start predicting...")
        predict_only(args)