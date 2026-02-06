import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
import re
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import websocket  # 使用websocket_client
from PIL import Image
import io


# appid = "73e0ee7a"    #填写控制台中获取的 APPID 信息
# api_secret = "NjZkYTlhOGViMjNiNjIwZjFkNGVmNmY4"   #填写控制台中获取的 APISecret 信息
# api_key ="2cc62bbfcbdd247535b166809cfd0382"    #填写控制台中获取的 APIKey 信息
appid = "2d306829"    #填写控制台中获取的 APPID 信息
api_secret = "YmZlMDBlNDg0N2U5MDRmZWI0YTAwZDA3"   #填写控制台中获取的 APISecret 信息
api_key ="182d20e7dc3c166477f90d1fd627bac3"
# imagedata = open("/root/dataset/Objaverse-2D/02378c56244e43f18c09d72cbb40a7db/rgb_000_front.png",'rb').read()

def combine_images_to_bytes(byte_data_list):
    """
    输入: 6个视角图片的 bytes 列表
    输出: 拼接后 2x3 大图的 bytes 数据
    """
    img_size = 512
    rows, cols = 2, 3

    # 1. 在内存中完成拼接
    canvas = Image.new('RGB', (img_size * cols, img_size * rows))

    for index, raw_data in enumerate(byte_data_list):
        img = Image.open(io.BytesIO(raw_data))
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size))

        x = (index % cols) * img_size
        y = (index // cols) * img_size
        canvas.paste(img, (x, y))

    # 2. 将拼接后的 PIL 对象转回 bytes
    img_byte_arr = io.BytesIO()
    canvas.save(img_byte_arr, format='PNG')

    final_imagedata = img_byte_arr.getvalue()

    return final_imagedata

# 1. 准备路径和视角顺序
base_path = "/root/dataset/Objaverse-2D/02378c56244e43f18c09d72cbb40a7db/"
view_order = ['front_left', 'front', 'front_right', 'left', 'back', 'right']

# 2. 读取二进制数据并存入列表
all_byte_data = []
for view in view_order:
    path = f"{base_path}rgb_000_{view}.png"
    with open(path, 'rb') as f:
        all_byte_data.append(f.read())

# 3. 调用函数获取拼接后的对象
imagedata = combine_images_to_bytes(all_byte_data)

imageunderstanding_url = "wss://spark-api.cn-huabei-1.xf-yun.com/v2.1/image"#云端环境的服务地址
text =[{"role": "user", "content": str(base64.b64encode(imagedata), 'utf-8'), "content_type":"image"}]

class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, imageunderstanding_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(imageunderstanding_url).netloc
        self.path = urlparse(imageunderstanding_url).path
        self.ImageUnderstanding_url = imageunderstanding_url

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.ImageUnderstanding_url + '?' + urlencode(v)
        #print(url)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url

# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)

# 收到websocket关闭的处理
def on_close(ws,one,two):
    print(" ")

# 收到websocket连接建立的处理
def on_open(ws):
    thread.start_new_thread(run, (ws,))

def run(ws, *args):
    data = json.dumps(gen_params(appid=ws.appid, question= ws.question ))
    ws.send(data)

# 收到websocket消息的处理
def on_message(ws, message):
    #print(message)
    data = json.loads(message)
    code = data['header']['code']
    if code != 0:
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]
        print(content,end ="")
        global answer
        answer += content
        # print(1)
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

def main(appid, api_key, api_secret, imageunderstanding_url,question):

    wsParam = Ws_Param(appid, api_key, api_secret, imageunderstanding_url)
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.appid = appid
    #ws.imagedata = imagedata
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
    #print("text-content-tokens:", getlength(text[1:]))
    while (getlength(text[1:])> 8000):
        del text[1]
    return text

if __name__ == '__main__':
    # Input = "Given an image, describe its overall global of the object and the 3D relationship between components, and its local features using short phrases."
    Input = "This image is a 2×3 grid containing six multi-view photos of a single object. The layout is:Top Row (Left to Right): Front-Left, Front, Front-RightBottom Row (Left to Right): Left, Back, RightTask:Global Structure: Describe the object's overall geometry and topology by synthesizing information from all six views, using short phrases.3D Spatial Relationships: Explain how components connect and relate in 3D space, using short phrases.View-Specific Local Features: Identify unique textures or parts visible in specific views, using short phrases.Aesthetic Evaluation: Assess the aesthetic quality (1-10) silently. Do not write any explanation, reasoning, or summary for the score. The very last line of your response must strictly follow this exact format:AESTHETIC_SCORE: X(Where X is the integer score)."
    question = checklen(getText("user",Input))
    answer = ""
    print("答:",end = "")
    main(appid, api_key, api_secret, imageunderstanding_url, question)
    response_text = getText("assistant", answer)
    # 提取评分
    score_match = re.search(r'AESTHETIC_SCORE: (\d+)', response_text[2].get("content"))
    aesthetic_score = int(score_match.group(1)) if score_match else None # socre

