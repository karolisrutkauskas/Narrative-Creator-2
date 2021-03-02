import sys
import json

files_from = int(sys.argv[1])
files_to = int(sys.argv[2])
read_file = sys.argv[3]
mode = sys.argv[4]

list_of_ids = []
with open(read_file, 'r') as jsonl_file:
    json_list = list(jsonl_file)
    for item in json_list[files_from:files_to]:
        result = json.loads(item)
        list_of_ids.append(mode + '/' + result['image_id'])

print(list_of_ids)

with open('data/file_ids.txt', 'a') as txt_file:
    for line in list_of_ids:
        txt_file.write(line)
        txt_file.write('\n')