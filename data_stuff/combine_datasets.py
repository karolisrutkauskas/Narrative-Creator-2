import sys
import os
import json

def get_image_ids(image_dir):
    images = os.listdir(image_dir)
    images_no_ext = []
    for image in images:
        img_name_len = len(image) - 4
        images_no_ext.append(image[:img_name_len])
    return images_no_ext

image_dir = sys.argv[1]
jsonl_dir = sys.argv[2]
out = sys.argv[3]

image_ids = get_image_ids(image_dir)

with open(jsonl_dir, 'r') as jsonl_file:
    json_list = list(jsonl_file)
    narrative_list = {}
    for item in json_list:
        result = json.loads(item)
        narrative_list[result['image_id']] = result['caption']

data_pairs = {}
for image_id in image_ids:
    data_pairs[image_id] = narrative_list[image_id]

with open(out, 'a') as jsonl_file:
    for key in data_pairs:
        print(data_pairs[key])
        if len(data_pairs[key]) != 0:
            data = {'image_id': key, 'narrative': data_pairs[key]}
            jsonl_file.write(json.dumps(data))
            jsonl_file.write('\n')