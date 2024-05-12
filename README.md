# data-clustering
a simple repository for unsupervised clustering based on sentence embeddings

## Quick Start
The main pipe of our repository is `prepare` --> `clustering` --> `sampling`.

We regard sentence embeddings as the semantic representation of the content that needs to be clustered. We use [angle-llama-7b-nli-v2](https://huggingface.co/SeanLee97/angle-llama-7b-nli-v2) in our codes.
```bash
bash prepare.sh
```
After obtaining the vectors, we select some (or all) of the vectors to train our clusterer.
```bash
bash clustering.sh
```
Specifically, we support two type of clusterer (K-Means and DBSCAN).
You need to install any library among cuml, sklearn, and faiss to perform clustering. We recommend [cuml](https://github.com/rapidsai/cuml) because it supports GPU computing.
(You can use `post_process.py` to check the results of clustering. It will print the center data points for each cluster.)

ps: You can use argument `--predict-only` to predict data using the trained model.

Finally, we sample the clustered data according to categories, and set the total sampling amount of different categories (allocated according to the actual amount of data in each category) for random sampling. The code is in `balance_sampling.py`.
