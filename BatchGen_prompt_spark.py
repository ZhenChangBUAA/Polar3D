import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
import re
import os
import time
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import websocket  # 使用websocket_client
from PIL import Image
import io

# [0]
appid = "80a8832d"
api_secret = "ZWJjN2YwMzRiYWZiODMwZThiM2FlMjU5"
api_key = "fd83c7cea4973c19511b54788aa483be"

# [1]
# appid = "38832ad9"
# api_secret ="NmIwYTU1Njk3NGUwZjIzYWJjOGJiYzhi"
# api_key ="b98a4ed73699911c1a79e954e9607f3e"

# [2]
# appid = "fea78205"
# api_secret ="ODUyN2IwYThiMGRkMGY3ODIyYzM0MmVk"
# api_key ="51663c4d0495ebbe41c8fee4ba093f7c"
#[3]
# appid = "73e0ee7a"
# api_secret = "NjZkYTlhOGViMjNiNjIwZjFkNGVmNmY4"
# api_key = "2cc62bbfcbdd247535b166809cfd0382"


imageunderstanding_url = "wss://spark-api.cn-huabei-1.xf-yun.com/v2.1/image"

text = []
answer = ""

# -------------------------- 数据分组 --------------------------
# 文件路径
# TOTAL_UID_FILE = "/home/beihang/mhl/Wonder3D-plus/dataset_all/folder_names.json"
# DONE_UID_FILE = "/home/beihang/mhl/Wonder3D-plus/dataset_all/havedone.json"

TOTAL_UID_FILE = "/home/beihang/mhl/Wonder3D-plus/dataset_all/folder_names.json"
DONE_UID_FILE= "/home/beihang/mhl/Wonder3D-plus/dataset_all/havedone.json"
# 每组数据量（每个API处理1000条）
GROUP_SIZE = 1000
# -------------------------- 分组函数--------------------------
def load_json_file(file_path):
    """
    加载JSON文件，处理文件不存在、格式错误等异常
    """
    # 如果文件不存在，返回空列表
    if not os.path.exists(file_path):
        print(f"提示：{file_path} 文件不存在，将返回空列表")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 校验数据格式（必须是列表）
        if not isinstance(data, list):
            raise ValueError("文件内容不是合法的列表格式")
        return data
    except json.JSONDecodeError:
        print(f"错误：{file_path} JSON格式损坏，无法加载")
        return []
    except Exception as e:
        print(f"错误：加载 {file_path} 失败 - {e}")
        return []
def get_undone_uid_list(total_uids, done_uids):
    """
    计算未处理的UID列表（总列表 - 已处理列表）
    转为集合去重，避免总列表或已处理列表中存在重复UID
    """
    # 转为集合去重并求差集
    total_uid_set = set(total_uids)
    done_uid_set = set(done_uids)
    undone_uid_set = total_uid_set - done_uid_set

    # 转回列表并排序（保证顺序稳定，方便分组）
    undone_uids = sorted(list(undone_uid_set))
    print(f"数据统计：")
    print(f"  总UID数量：{len(total_uid_set)}（去重后）")
    print(f"  已处理UID数量：{len(done_uid_set)}（去重后）")
    print(f"  未处理UID数量：{len(undone_uids)}")

    return undone_uids
def split_uids_into_groups(undone_uids, group_size=GROUP_SIZE):
    """
    将未处理UID列表按指定大小分组
    返回：分组后的列表（每个元素是一个包含1000个UID的子列表）
    """
    groups = []
    total_undone = len(undone_uids)

    # 分片分组（步长为group_size）
    for i in range(0, total_undone, group_size):
        end_idx = min(i + group_size, total_undone)
        group = undone_uids[i:end_idx]
        groups.append(group)

    print(f"分组完成：共分为 {len(groups)} 组，每组最多 {group_size} 条数据")
    for idx, group in enumerate(groups):
        print(f"  第 {idx + 1} 组：{len(group)} 条UID")

    return groups

def combine_images_to_bytes(byte_data_list):

    img_size = 512
    rows, cols = 2, 3

    if not byte_data_list:
        return None

    canvas = Image.new('RGB', (img_size * cols, img_size * rows))

    for index, raw_data in enumerate(byte_data_list):
        try:
            img = Image.open(io.BytesIO(raw_data))
            if img.size != (img_size, img_size):
                img = img.resize((img_size, img_size))

            x = (index % cols) * img_size
            y = (index // cols) * img_size
            canvas.paste(img, (x, y))
        except Exception as e:
            print(f"Error processing image {index}: {e}")

    img_byte_arr = io.BytesIO()
    canvas.save(img_byte_arr, format='JPEG', quality=90)

    final_imagedata = img_byte_arr.getvalue()

    return final_imagedata

class Ws_Param(object):
    def __init__(self, APPID, APIKey, APISecret, imageunderstanding_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(imageunderstanding_url).netloc
        self.path = urlparse(imageunderstanding_url).path
        self.ImageUnderstanding_url = imageunderstanding_url

    def create_url(self):
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')
        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        url = self.ImageUnderstanding_url + '?' + urlencode(v)
        return url

def on_error(ws, error):
    print("### error:", error)

def on_close(ws, one, two):
    print(" ")

def on_open(ws):
    thread.start_new_thread(run, (ws,))

def run(ws, *args):
    data = json.dumps(gen_params(appid=ws.appid, question=ws.question))
    ws.send(data)

def on_message(ws, message):
    data = json.loads(message)
    code = data['header']['code']
    if code != 0:
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]
        # print(content,end ="") # ### [修改] 批量处理时可以注释掉实时打印，或者保留
        global answer
        answer += content
        if status == 2:
            ws.close()

def gen_params(appid, question):
    """
    通过appid和用户的提问来生成请参数
    """
    data = {
        "header": {
            "app_id": appid
        },
        "parameter": {
            "chat": {
                "domain": "imagev3",
                "temperature": 0.5,
                "top_k": 4,
                "max_tokens": 2028,
                "auditing": "default"
            }
        },
        "payload": {
            "message": {
                "text": question
            }
        }
    }
    return data

def main(appid, api_key, api_secret, imageunderstanding_url, question):
    wsParam = Ws_Param(appid, api_key, api_secret, imageunderstanding_url)
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.appid = appid
    ws.question = question
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

def getText(role, content):
    jsoncon = {}
    jsoncon["role"] = role
    jsoncon["content"] = content
    text.append(jsoncon)
    return text

def getlength(text):
    length = 0
    for content in text:
        temp = content["content"]
        leng = len(temp)
        length += leng
    return length

def checklen(text):
    while (getlength(text[1:]) > 8000):
        del text[1]
    return text

def save_results_to_json(filepath, data):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"写入文件失败: {e}")

def batch_process_dataset(dataset_root, output_json_path,group):
    global text, answer

    results = []
    processed_ids = set()

    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content:  # 防止文件为空
                    results = json.loads(content)
                    # 提取已处理的 folder_id
                    processed_ids = {item['folder_id'] for item in results}
            print(f"检测到已有记录，已加载 {len(results)} 条数据，将跳过这些项目。")
        except json.JSONDecodeError:
            print("警告：JSON 文件格式错误或损坏，将重新开始记录。")
            results = []

    # 获取根目录下所有子文件夹
    all_folders = [f for f in group if os.path.isdir(os.path.join(dataset_root, f))]
    total_folders = len(all_folders)

    view_order = ['front_left', 'front', 'front_right', 'left', 'back', 'right']

    # 提示词
    prompt_input = "This image is a 2×3 grid containing six multi-view photos of a single object. The layout is: Top Row (Left to Right): Front-Left, Front, Front-Right. Bottom Row (Left to Right): Left, Back, Right. Task: Global Structure: Describe the object's overall geometry and topology by synthesizing information from all six views, using short phrases. 3D Spatial Relationships: Explain how components connect and relate in 3D space, using short phrases. View-Specific Local Features: Identify unique textures or parts visible in specific views, using short phrases. Aesthetic Evaluation: Assess the aesthetic quality (1-10) silently. Do not write any explanation, reasoning, or summary for the score. The very last line of your response must strictly follow this exact format: AESTHETIC_SCORE: X (Where X is the integer score)."

    for idx, folder_id in enumerate(all_folders):
        print(f"[{idx + 1}/{total_folders}] 正在检查: {folder_id}", end=" ... ")

        if folder_id in processed_ids:
            print("跳过 (已在 JSON 中)")
            continue

        print("开始处理")
        current_folder_path = os.path.join(dataset_root, folder_id)

        # 2. 读取图片
        all_byte_data = []
        is_files_complete = True
        for view in view_order:
            img_path = os.path.join(current_folder_path, f"rgb_000_{view}.png")
            if not os.path.exists(img_path):
                print(f"  警告: 缺失文件 {img_path}，跳过此物体。")
                is_files_complete = False
                break
            with open(img_path, 'rb') as f:
                all_byte_data.append(f.read())

        if not is_files_complete:
            continue

        # 3. 拼接图片
        imagedata = combine_images_to_bytes(all_byte_data)

        # 重置上下文
        text = []
        answer = ""

        text.append({"role": "user", "content": str(base64.b64encode(imagedata), 'utf-8'), "content_type": "image"})
        question = checklen(getText("user", prompt_input))

        # 4. 调用 API
        try:
            main(appid, api_key, api_secret, imageunderstanding_url, question)

            response_content = answer

            # 解析
            score_match = re.search(r'AESTHETIC_SCORE:\s*(\d+)', response_content)
            aesthetic_score = int(score_match.group(1)) if score_match else None
            description = re.sub(r'AESTHETIC_SCORE:\s*\d+', '', response_content).strip()

            # 存入结果列表
            item_result = {
                "folder_id": folder_id,
                "description": description,
                "aesthetic_score": aesthetic_score
            }
            results.append(item_result)
            processed_ids.add(folder_id)  # 标记为已处理
            # 将 description 单独保存为 txt 文件
            caption_path = os.path.join(current_folder_path, "detail_caption.txt")
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(description)


            # ### 实时写入文件
            save_results_to_json(output_json_path, results)

        except Exception as e:
            print(f"  API调用或解析出错: {e}")

        time.sleep(0.5)

    print(f"全部任务结束，最终结果保存在 {output_json_path}")

if __name__ == '__main__':

    # ### [修改] 改为定义数据集路径和输出路径
    dataset_dir = "/home/beihang/mhl/Wonder3D-plus/dataset_all/"
    output_file = "/home/beihang/mhl/Wonder3D-plus/dataset_all/objaverse_aesthetic_scores.json"

    total_uids = load_json_file(TOTAL_UID_FILE)
    done_uids = load_json_file(DONE_UID_FILE)
    # 2. 获取未处理UID列表
    undone_uids = get_undone_uid_list(total_uids, done_uids)
    # 3. 分组生成（核心：创建分组变量，方便你分配给不同API）
    uid_groups = split_uids_into_groups(undone_uids, GROUP_SIZE)

    # 23个分组

    batch_process_dataset(dataset_dir, output_file,uid_groups[0])