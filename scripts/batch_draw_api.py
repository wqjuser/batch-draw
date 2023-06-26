from fastapi import Depends, FastAPI, HTTPException, Request
import asyncio
from typing import Any, List, Optional, TypedDict
from pydantic import BaseModel
from revChatGPT.V1 import Chatbot as ChatbotV1
from revChatGPT.V3 import Chatbot as ChatbotV3
from modules import script_callbacks
import copy
import hashlib
import importlib.util
import json
import os
import random
import re
import shlex
import string
import subprocess
import sys
import threading
import traceback
from datetime import datetime

import gradio as gr
import requests
from PIL import Image, ImageDraw, ImageFont
from gotrue.errors import AuthApiError
from natsort import natsorted
from openai import APIError

import modules.scripts as scripts
from modules import images
from modules.processing import process_images
from modules.shared import state
from scripts import prompts_styles as ps
from scripts import voice_params as vop
from scripts import batch_draw_utils
from revChatGPT.V1 import Chatbot as ChatbotV1
from revChatGPT.V3 import Chatbot as ChatbotV3
from dotenv import load_dotenv, set_key
import uuid
from urllib.parse import quote_plus
from urllib.parse import urlencode
import time
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
import base64
import wave
import azure.cognitiveservices.speech as speechsdk
from supabase import create_client, Client
from datetime import datetime, timedelta, timezone
import ntplib
from time import ctime

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
env_path = f"{parent_dir}\\.env" if sys.platform == 'win32' else f"{parent_dir}/.env"
load_dotenv(dotenv_path=env_path, override=True, verbose=True)
mac = uuid.getnode()
mac_address = ':'.join(("%012X" % mac)[i:i + 2] for i in range(0, 12, 2))
machine_code = batch_draw_utils.get_unique_machine_code()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)
current_date = datetime.now()
formatted_date = current_date.strftime("%Y-%m-%d")
novel_tweets_generator_images_folder = "outputs/novel_tweets_generator/images"
os.makedirs(novel_tweets_generator_images_folder, exist_ok=True)
novel_tweets_generator_prompts_folder = "outputs/novel_tweets_generator/prompts"
os.makedirs(novel_tweets_generator_prompts_folder, exist_ok=True)
novel_tweets_generator_audio_folder = "outputs/novel_tweets_generator/audio"
os.makedirs(novel_tweets_generator_audio_folder, exist_ok=True)


def chang_time_zone(utc_time_str):
    utc_time = datetime.fromisoformat(utc_time_str)
    east_eight_zone = timezone(timedelta(hours=8))
    local_time = utc_time.astimezone(east_eight_zone)
    local_time_str = local_time.strftime("%Y-%m-%d %H:%M:%S")
    return local_time_str


def get_ntp_time():
    ntp_client = ntplib.NTPClient()
    response = ntp_client.request('pool.ntp.org')
    return response.tx_time


def compare_time(time_str):
    date_format = "%Y-%m-%d %H:%M:%S"
    timestamp = time.mktime(time.strptime(time_str, date_format))
    now = time.time()
    if timestamp > now:
        return False
    elif timestamp < now:
        return True


def call_rpc_function(param1: str, param2: str, param3: str):
    response = supabase.rpc('get_env_by_user_v6', {
        'param_user_id': param1,
        'param_active_code': param2,
        'param_mac_address': param3
    }).execute()
    return response


userid = os.environ.get("USER_ID")
active_code = os.environ.get("ACTIVE_CODE")
realtime = ''
is_expired = True
env_data = {}
default_response = {
    'code': 200,
    'data': {},
}
if userid != '' and active_code != '':
    res_data = call_rpc_function(userid, active_code, machine_code)
    try:
        if res_data.data[0]['code'] == 0:
            env_data = res_data.data[0]['data']['env']
            env_expire_time = res_data.data[0]['data']['expire_at']
            realtime = chang_time_zone(env_expire_time)
            set_key(env_path, 'EXPIRE_AT', realtime)
            os.environ['CHATGPT_BASE_URL'] = env_data['CHATGPT_BASE_URL']
            os.environ['API_URL'] = env_data['API_URL']
            is_expired = compare_time(realtime)
            set_key(env_path, 'ACTIVE_INFO', f'脚本已激活，到期时间是:{realtime}，在此期间祝你玩的愉快。')
    except APIError as e:
        print("获取环境配置失败，请重启webui")


def batch_draw_api(_: Any, app: FastAPI):
    pre = "/batch_draw"

    @app.get(f"{pre}/welcome")
    async def greeting():
        default_response['data']['result'] = '欢迎使用batch_draw_api'
        return default_response

    class GenScenePromptParams(BaseModel):
        original_article: str
        gen_type: int = 0
        use_proxy: bool = False
        user_pre_prompt: str = ''
        scene_number: int = 10

    @app.post(f"{pre}/gen_scene_prompt")
    async def gen_scene_prompt(req: GenScenePromptParams):
        default_pre_prompt = """你是专业的场景分镜描述专家，我给你一段文字，首先你需要将文字内容改得更加吸引人，然后你需要把修改后的文字分为不同的场景分镜。每个场景必须要细化，要给出人物，时间，地点，场景的描述，如果分镜不存在人物就写无人。必须要细化环境描写（天气，周围有些什么等等内容），必须要细化人物描写（人物衣服，衣服样式，衣服颜色，表情，动作，头发，发色等等），如果多个分镜中出现的人物是同一个，请统一这个人物的衣服，发色等细节。如果分镜中出现多个人物，还必须要细化每个人物的细节。
你回答的分镜要加入自己的一些想象，但不能脱离原文太远。你的回答请务必将每个场景的描述转换为单词，并使用多个单词描述场景，每个分镜至少6个单词，如果分镜中出现了人物,请给我添加人物数量的描述。
你还需要分析场景分镜中各个物体的比重并且将比重按照提示的格式放在每个单词的后面。你只用回复场景分镜内容，其他的不要回复。
例如这一段话：我和袁绍是大学的时候认识的，在一起了三年。毕业的时候袁绍说带我去他家见他爸妈。去之前袁绍说他爸妈很注重礼节。还说别让我太破费。我懂，我都懂......于是我提前去了我表哥顾朝澜的酒庄随手拿了几瓶红酒。临走我妈又让我再带几个LV的包包过去，他妈妈应该会喜欢的。我也没多拿就带了两个包，其中一个还是全球限量版。女人哪有不喜欢包的，所以我猜袁绍妈妈应该会很开心吧。
将它分为四个场景，你需要这样回答我：
1. 情侣, (一个女孩和一个男孩:1.5), (女孩黑色的长发:1.2), 微笑, (白色的裙子:1.2), 非常漂亮的面庞, (女孩手挽着一个男孩:1.5), 男孩黑色的短发, (穿着灰色运动装, 帅气的脸庞:1.2), 走在大学校园里, 
2. 餐馆内，一个女孩, (黑色的长发, 白色的裙子:1.5), 坐在餐桌前, 一个男孩坐在女孩的对面, (黑色的短发, 灰色的外套:1.5), 两个人聊天.
3. 酒庄内，一个女孩，微笑，(黑色的长发，白色的裙子:1.2)，(站着:1.5)，(拿着1瓶红酒:1.5)
4. 一个女孩，(白色的裙子，黑色的长发:1.5)，(手上拿着两个包:1.5)，站在豪华的客厅内，
不要拘泥于我给你示例中的权重数字，权重的范围在1到2之前的权重值。你需要按照分镜中的画面自己判断权重。注意回复中的所有标点符号请使用英文的标点符号包括逗号，不要出现句号，请你牢记这些规则，任何时候都不要忘记。
"""
        if req.user_pre_prompt != '':
            default_pre_prompt = req.user_pre_prompt
        if int(req.scene_number) != 0:
            prompt = default_pre_prompt + "\n" + f"内容是：{req.original_article}\n必须将其转换为{int(req.scene_number)}个场景分镜。"
        else:
            prompt = default_pre_prompt + "\n" + f"内容是：{req.original_article}\n你需要根据文字内容自己分析可以转换成几个场景分镜。"
        if req.use_proxy:
            proxy = os.environ.get('PROXY')
        else:
            proxy = ''
        if req.gen_type == 0:
            try:
                openai_key = env_data['KEY']
                chatbot = ChatbotV3(api_key=openai_key, proxy=proxy if (proxy != "" or proxy is not None) else None)
                response = chatbot.ask(prompt=prompt)
            except Exception as error:
                print(f"Error: {error}")
                response = "抱歉，发生了一些意外，请重试。"
                default_response['code'] = 501
                default_response['data']['result'] = response
        else:
            configs = {
                "access_token": f"{env_data['ACCESS_TOKEN']}",
                "disable_history": True
            }
            if proxy is not None and proxy != "":
                configs['proxy'] = proxy.replace('http://', '')
            try:
                chatbot = ChatbotV1(config=configs)
                for data in chatbot.ask(prompt):
                    response = data["message"]
                    default_response['data']['result'] = response
            except Exception as error:
                print(f"Error: {error}")
                response = "抱歉，发生了一些意外，请重试。"
                default_response['code'] = 502
                default_response['data']['result'] = response
        return default_response

    @app.get(f"{pre}/ask_ai")
    async def ask_ai(prompt: str, gen_type: int = 0):
        if gen_type == 0:
            try:
                openai_key = env_data['KEY']
                chatbot = ChatbotV3(api_key=openai_key)
                response = chatbot.ask(prompt=prompt)
                default_response['data']['result'] = response
            except Exception as error:
                print(f"Error: {error}")
                response = "抱歉，发生了一些意外，请重试。"
                default_response['code'] = 501
                default_response['data']['result'] = response
        else:
            configs = {
                "access_token": f"{env_data['ACCESS_TOKEN']}",
                "disable_history": True
            }
            try:
                chatbot = ChatbotV1(config=configs)
                for data in chatbot.ask(prompt, auto_continue=True):
                    response = data["message"]
                    default_response['data']['result'] = response
            except Exception as error:
                print(f"Error: {error}")
                response = "抱歉，发生了一些意外，请重试。"
                default_response['code'] = 502
                default_response['data']['result'] = response
        return default_response

    @app.get(f"{pre}/signup")
    async def signup(code: str):
        global env_data
        random_str1 = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        random_str2 = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
        random_str3 = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        try:
            email = f"{random_str1}@{random_str2}.com"
            res = supabase.auth.sign_up({
                "email": email,
                "password": random_str3
            })
            res_dict = json.loads(res.json())
            if res_dict["user"]:
                user_id = res_dict["user"]["id"]
                res = call_rpc_function(user_id, code, machine_code)
                if res.data[0]['code'] == 0:
                    expire_time = chang_time_zone(res.data[0]['data']['expire_at'])
                    env_data = res.data[0]['data']['env']
                    set_key(env_path, 'USER_ID', user_id)
                    set_key(env_path, 'ACTIVE_CODE', code)
                    set_key(env_path, 'ACTIVE_INFO', f'脚本已激活，到期时间是:{expire_time}，在此期间祝你玩的愉快。')
                    os.environ['CHATGPT_BASE_URL'] = res.data[0]['data']['env']['CHATGPT_BASE_URL']
                    os.environ['API_URL'] = res.data[0]['data']['env']['API_URL']
                    response = '激活成功'
                    default_response['data']['result'] = response
                else:
                    response = f"激活异常:----->{res.data[0]['msg']}"
                    default_response['code'] = 501
                    default_response['data']['result'] = response
            else:
                response = f"用户注册失败"
                default_response['code'] = 502
                default_response['data']['result'] = response
        except APIError as error:
            response = f"用户注册失败----->{error}"
            default_response['code'] = 503
            default_response['data']['result'] = response
        except AuthApiError as error:
            response = f"用户注册失败----->{error}"
            default_response['code'] = 504
            default_response['data']['result'] = response
        return default_response

    @app.get(f"{pre}tts/baidu")
    async def tts_baidu(aue, per, pit, spd, text, vol):
        file_count = 0
        for root, dirs, files in os.walk(novel_tweets_generator_audio_folder):
            file_count += len(files)
        is_short = True
        if len(text) > 60:
            is_short = False
        if is_short:
            if aue == 'mp3-16k' or aue == 'mp3-48k':
                aue = 'mp3'
        else:
            if aue == 'mp3':
                aue = 'mp3-16k'
        audio_format = aue
        for i, role in enumerate(vop.baidu['voice_role']):
            if per == role:
                per = vop.baidu['role_number'][i]
        for i, out_type in enumerate(vop.baidu['aue']):
            if aue == out_type:
                aue = vop.baidu['aue_num'][i]
        data = {
            'grant_type': 'client_credentials',
            'client_id': env_data['BAIDU_VOICE_API_KEY'],
            'client_secret': env_data['BAIDU_VOICE_SECRET_KEY'],
        }
        tts_url = vop.baidu['short_voice_url'] if is_short else vop.baidu['long_voice_create_url']
        response = requests.post(vop.baidu['get_access_token_url'], data=data)
        if response.status_code == 200:
            response_dict = json.loads(response.text)
            access_token = response_dict['access_token']
            text_encode = quote_plus(text.encode('utf-8'))
            headers_short = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': '*/*'
            }
            headers_long = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            if is_short:
                payload = {
                    "lan": "zh",
                    "tex": text_encode,
                    "tok": access_token,
                    "cuid": mac_address,
                    "ctp": 1,
                    "spd": spd,
                    "pit": pit,
                    "vol": vol,
                    "per": per,
                    "aue": aue
                }
            else:
                payload = {
                    "lang": "zh",
                    "text": text,
                    "format": audio_format,
                    "voice": per,
                    "speed": spd,
                    "pitch": pit,
                    "volume": vol,
                }
            if not is_short:
                tts_url = f"{tts_url}?access_token={access_token}"
                data = json.dumps(payload)
                response1 = requests.post(tts_url, headers=headers_long, data=data)
            else:
                data = urlencode(payload)
                response1 = requests.post(tts_url, headers=headers_short, data=data.encode('utf-8'))
            if not is_short:
                if response1.status_code == 200:
                    response_dict = json.loads(response1.text)
                    if 'error_code' in response_dict:
                        return_response = f"百度长文本转语音任务创建失败，错误原因：{response_dict['error_msg']}"
                        print("百度长文本转语音任务创建失败，错误原因：", response_dict['error_msg'])
                    elif 'error' in response_dict:
                        print("百度长文本转语音任务创建失败，错误原因：", f"{response_dict['error']}---{response_dict['message']}")
                    else:
                        task_id = response_dict['task_id']
                        print("百度长文本转语音任务创建成功，任务完成后自动下载，你可以在此期间做其他的事情。")
                        payload = json.dumps({
                            'task_ids': [f"{task_id}"]
                        })
                        while True:
                            response2 = requests.post(f"{vop.baidu['long_voice_query_url']}?access_token={access_token}", headers=headers_long,
                                                      data=payload)
                            rj = response2.json()
                            if response2.status_code == 200:
                                if rj['tasks_info'][0]['task_status'] == 'Success':
                                    speech_url = rj['tasks_info'][0]['task_result']['speech_url']
                                    response3 = requests.get(speech_url)
                                    file_ext = audio_format
                                    if audio_format == 'pcm-8k' or audio_format == 'pcm-16k':
                                        file_ext = 'pcm'
                                    elif audio_format == 'mp3-16k' or audio_format == 'mp3-48k':
                                        file_ext = 'mp3'
                                    file_path = os.path.join(novel_tweets_generator_audio_folder, f'{file_count + 1}.{file_ext}')
                                    with open(file_path, 'wb') as f:
                                        f.write(response3.content)
                                    if sys.platform == 'win32':
                                        print("语音下载完成，保存路径是:----->", os.getcwd() + "\\" + file_path)
                                    else:
                                        print("语音下载完成，保存路径是:----->", os.getcwd() + "/" + file_path)
                                    break
                                elif rj['tasks_info'][0]['task_status'] == 'Running':
                                    time.sleep(10)
                                elif rj['tasks_info'][0]['task_status'] == 'Failure':
                                    print("百度长文本合成语音失败，原因是----->", f"{rj['tasks_info'][0]['task_result']['err_msg']}")
                                    break
                            else:
                                break
            file_ext_dict = {
                3: 'mp3',
                4: 'pcm',
                5: 'pcm',
                6: 'wav'
            }
            content_type_dict = {
                3: 'audio/mp3',
                4: 'audio/basic;codec=pcm;rate=16000;channel=1',
                5: 'audio/basic;codec=pcm;rate=8000;channel=1',
                6: 'audio/wav'
            }
            if is_short:
                content_type = response1.headers['Content-Type']
                if aue in file_ext_dict and content_type == content_type_dict[aue]:
                    file_ext = file_ext_dict[aue]
                    file_path = os.path.join(novel_tweets_generator_audio_folder, f'{file_count + 1}.{file_ext}')
                    with open(file_path, 'wb') as f:
                        f.write(response1.content)
                else:
                    print("百度短文本语音合成失败，请稍后重试")
        else:
            print('百度语音合成请求失败，请稍后重试')


try:
    script_callbacks.on_app_started(batch_draw_api)
    print('batch_draw background API service started successfully.')
except Exception as e:
    print(f'batch_draw background API service failed to start: {e}')
