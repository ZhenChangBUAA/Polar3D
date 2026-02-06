import json

def read_json_to_list(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        # json.load() 直接将JSON数组转为Python list
        json_list = json.load(f)
    return json_list



list1 = read_json_to_list("train_lora.json")  # 你的第一个JSON文件路径
list2 = read_json_to_list("train_runable.json")  # 你的第二个JSON文件路径

set1 = set(list1)
set2 = set(list2)
common = set1 & set2
# 输出结果
print(len(common))
for idx, item in enumerate(sorted(common), 1):
    print(f"{idx}. {item}")
print(f"第一个JSON文件转换后的列表长度：{len(list1)}")
print(f"第二个JSON文件转换后的列表长度：{len(list2)}")