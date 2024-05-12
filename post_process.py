import json
import jsonlines
from collections import defaultdict

def classify_and_save(input_file, index_output_file, info_output_file):
    class_info = defaultdict(list)
    class_indices = defaultdict(list)
    
    with jsonlines.open(input_file) as reader:
        for idx, obj in enumerate(reader):
            class_value = obj.get('class')
            class_indices[class_value].append(idx)
            class_info[class_value].append(obj)

    class_info_summary = []
    for class_id, (class_values, indices) in enumerate(class_indices.items(), start=1):
        class_info_summary.append({
            'id': class_id,
            'total_count': len(indices),
            'examples': class_info[class_values][:5]  # Take the first 5 examples
        })

    # Sort classes by total_count
    sorted_classes = sorted(class_info_summary, key=lambda x: x['total_count'])

    # Print top 10 classes with most counts
    print("Top 10 classes with most counts:")
    for item in sorted_classes[-10:][::-1]:
        print(f"Class ID: {item['id']}, Total Count: {item['total_count']}")

    # Print top 10 classes with least counts
    print("\nTop 10 classes with least counts:")
    for item in sorted_classes[:10]:
        print(f"Class ID: {item['id']}, Total Count: {item['total_count']}")

    with open(index_output_file, 'w') as index_file:
        json.dump(class_indices, index_file)

    with open(info_output_file, 'w') as info_file:
        json.dump(class_info_summary, info_file, indent=2)

if __name__ == "__main__":
    input_file = "label/results_all_final.jsonl"
    index_output_file = "label/results_all_final_index.jsonl"
    info_output_file = "label/results_all_final_info.jsonl"

    classify_and_save(input_file, index_output_file, info_output_file)
