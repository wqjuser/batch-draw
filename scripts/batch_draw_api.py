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

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
env_path = f"{parent_dir}\\.env" if sys.platform == 'win32' else f"{parent_dir}/.env"
load_dotenv(dotenv_path=env_path, override=True, verbose=True)
mac = uuid.getnode()
mac_address = ':'.join(("%012X" % mac)[i:i + 2] for i in range(0, 12, 2))
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)


def chang_time_zone(utc_time_str):
    utc_time = datetime.fromisoformat(utc_time_str)
    east_eight_zone = timezone(timedelta(hours=8))
    local_time = utc_time.astimezone(east_eight_zone)
    local_time_str = local_time.strftime("%Y-%m-%d %H:%M:%S")
    return local_time_str


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
if userid != '' and active_code != '':
    res_data = call_rpc_function(userid, active_code, mac_address)
    try:
        env_data = res_data.data[0]['data']['env']
        if res_data.data[0]['code'] == 0:
            env_expire_time = res_data.data[0]['data']['expire_at']
            realtime = chang_time_zone(env_expire_time)
            set_key(env_path, 'EXPIRE_AT', realtime)
            is_expired = compare_time(realtime)
            set_key(env_path, 'ACTIVE_INFO', f'脚本已激活，到期时间是:{realtime}，在此期间祝你玩的愉快。')
    except APIError as e:
        print("获取环境配置失败，请重启webui")
    except IndexError as e:
        print("获取环境配置失败，请重启webui")


def batch_draw_api(_: Any, app: FastAPI):
    pre = "/batch_draw"

    @app.get(f"{pre}/welcome")
    async def greeting():
        return "欢迎使用batch_draw_api"

    @app.get(f"{pre}/gen_scene_prompt")
    async def gen_scene_prompt(original_article: str, gen_type: int = 0, use_proxy: bool = False, user_pre_prompt: str = '', scene_number: int = 10):
        default_pre_prompt = """你是专业的场景分镜描述专家，我给你一段文字，并指定你需要转换的场景分镜个数，你需要把他分为不同的场景。每个场景必须要细化，要给出人物，时间，地点，场景的描述，如果分镜不存在人物就写无人。必须要细化环境描写（天气，周围有些什么等等内容），必须要细化人物描写（人物衣服，衣服样式，衣服颜色，表情，动作，头发，发色等等），如果多个分镜中出现的人物是同一个，请统一这个人物的衣服，发色等细节。如果分镜中出现多个人物，还必须要细化每个人物的细节。
        你回答的分镜要加入自己的一些想象，但不能脱离原文太远。你的回答请务必将每个场景的描述转换为单词，并使用多个单词描述场景，每个分镜至少6个单词，如果分镜中出现了人物,请给我添加人物数量的描述。
        你只用回复场景分镜内容，其他的不要回复。
        例如这一段话：我和袁绍是大学的时候认识的，在一起了三年。毕业的时候袁绍说带我去他家见他爸妈。去之前袁绍说他爸妈很注重礼节。还说别让我太破费。我懂，我都懂......于是我提前去了我表哥顾朝澜的酒庄随手拿了几瓶红酒。临走我妈又让我再带几个LV的包包过去，他妈妈应该会喜欢的。我也没多拿就带了两个包，其中一个还是全球限量版。女人哪有不喜欢包的，所以我猜袁绍妈妈应该会很开心吧。
        将它分为4个场景，你需要这样回答我：
        1. 情侣, 一个女孩和一个男孩, 女孩黑色的长发, 微笑, 白色的裙子, 非常漂亮的面庞, 女孩手挽着一个男孩, 男孩黑色的短发, 穿着灰色运动装, 帅气的脸庞, 走在大学校园里, 
        2. 餐馆内，一个女孩, 黑色的长发, 白色的裙子, 坐在餐桌前, 一个男孩坐在女孩的对面, 黑色的短发, 灰色的外套, 两个人聊天.
        3. 酒庄内，一个女孩，微笑，黑色的长发，白色的裙子，站着，拿着几瓶红酒，
        4. 一个女孩，白色的裙子，黑色的长发，手上拿着两个包，站在豪华的客厅内，
        请你牢记这些规则，任何时候都不要忘记。
            """
        if user_pre_prompt != '':
            default_pre_prompt = user_pre_prompt
        if int(scene_number) != 0:
            prompt = default_pre_prompt + "\n" + f"内容是：{original_article}\n必须将其转换为{int(scene_number)}个场景分镜。"
        else:
            prompt = default_pre_prompt + "\n" + f"内容是：{original_article}\n你需要根据文字内容自己分析可以转换成几个场景分镜。"
        response = ""
        if use_proxy:
            proxy = os.environ.get('PROXY')
        else:
            proxy = ''
        if gen_type == 0:
            try:
                openai_key = env_data['KEY']
                chatbot = ChatbotV3(api_key=openai_key, proxy=proxy if (proxy != "" or proxy is not None) else None)
                response = chatbot.ask(prompt=prompt)
            except Exception as error:
                print(f"Error: {error}")
                response = "抱歉，发生了一些意外，请重试。"
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
            except Exception as error:
                print(f"Error: {error}")
                response = "抱歉，发生了一些意外，请重试。"
        return response


try:
    script_callbacks.on_app_started(batch_draw_api)
    print('batch_draw background API service started successfully.')
except Exception as e:
    print(f'batch_draw background API service failed to start: {e}')
