# DONE
export CUDA_VISIBLE_DEVICES=0
NUM_CLUSTERS=1000;
TRAINING_SAMPLES=500000
DATASET=sft_100k;
QUERY_PATH=path_of_query;
SOURCE_PATH=path_of_source;
VECS_PATH=path_of_vecs;
LABELED_FILE_PATH=label/results_0_4000000.jsonl;
CLUSTERER_DIR=cluster/kmeans;
CLUSTERER_VERSION=step1000_data500k;
MAX_ITER=1000;
EPS=0.1;
SAMPLES=30;
CLUSTERER=kmeans;
START=0;
END=4000000;

python clustering.py \
--query-path ${QUERY_PATH} \
--vec-path ${VECS_PATH} \
--num-clusters ${NUM_CLUSTERS} \
--training-samples ${TRAINING_SAMPLES} \
--output-dir ${CLUSTERER_DIR}/${DATASET}/${CLUSTERER_VERSION}/ \
--max-iter ${MAX_ITER} \
--clusterer ${CLUSTERER} \
--eps ${EPS} \
--min-samples ${SAMPLES} \
--source-path ${SOURCE_PATH} \
--labeled-file-path ${LABELED_FILE_PATH} \
--start ${START} \
--end ${END} \
--has-vec \
# --predict-only \