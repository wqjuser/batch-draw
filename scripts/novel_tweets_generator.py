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
import traceback
import argparse
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
from scripts import preset_character as pc
from revChatGPT.V1 import Chatbot as ChatbotV1
from revChatGPT.V3 import Chatbot as ChatbotV3
from dotenv import load_dotenv, set_key
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


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag == "true") else False


def chang_time_zone(utc_time_str):
    utc_time = datetime.fromisoformat(utc_time_str)
    east_eight_zone = timezone(timedelta(hours=8))
    local_time = utc_time.astimezone(east_eight_zone)
    local_time_str = local_time.strftime("%Y-%m-%d %H:%M:%S")
    return local_time_str


def get_time_now():
    ntp_client = ntplib.NTPClient()
    local_time = time.time()
    try:
        response = ntp_client.request('pool.ntp.org')
        return response.tx_time
    except Exception as e:
        print(f"Error: {e}")
        return local_time


def compare_time(time_str):
    date_format = "%Y-%m-%d %H:%M:%S"
    timestamp = time.mktime(time.strptime(time_str, date_format))
    now = get_time_now()
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


prompt_tags = {
    "sd_model": None,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "denoising_strength": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag
}
current_date = datetime.now()
formatted_date = current_date.strftime("%Y-%m-%d")
novel_tweets_generator_images_folder = "outputs/novel_tweets_generator/images"
novel_tweets_generator_prompts_folder = "outputs/novel_tweets_generator/prompts"
novel_tweets_generator_audio_folder = "outputs/novel_tweets_generator/audio"
os.makedirs(novel_tweets_generator_prompts_folder, exist_ok=True)
os.makedirs(novel_tweets_generator_images_folder, exist_ok=True)
os.makedirs(novel_tweets_generator_audio_folder, exist_ok=True)
if sys.platform == 'win32':
    novel_tweets_generator_prompts_folder.replace('/', '\\')
    novel_tweets_generator_audio_folder.replace('/', '\\')
    novel_tweets_generator_images_folder.replace('/', '\\')
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
env_path = f"{parent_dir}\\.env" if sys.platform == 'win32' else f"{parent_dir}/.env"
load_dotenv(dotenv_path=env_path, override=True, verbose=True)
machine_code = batch_draw_utils.get_unique_machine_code()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)
userid = os.environ.get("USER_ID")
active_code = os.environ.get("ACTIVE_CODE")
realtime = ''
is_expired = True
env_data = {}
instructions = 'https://ixtrs1l7r3f.feishu.cn/docx/YN07dMSAXoh4eAxmviKcpAVdnrf'


def sign_up(code):
    global env_data
    random_str1 = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    random_str2 = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
    random_str3 = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    try:
        if os.environ.get('USER_ID') == '':
            email = f"{random_str1}@{random_str2}.com"
            res = supabase.auth.sign_up({
                "email": email,
                "password": random_str3
            })
            res_dict = json.loads(res.json())
        else:
            res_dict = {'user': {'id': os.environ.get('USER_ID')}}
        if res_dict["user"]:
            user_id = res_dict["user"]["id"]
            res = call_rpc_function(user_id, code, machine_code)
            if res.data[0]['code'] == 0:
                expire_time = chang_time_zone(res.data[0]['data']['expire_at'])
                set_key(env_path, 'EXPIRE_AT', expire_time)
                env_data = res.data[0]['data']['env']
                env_data['EXPIRE_AT'] = expire_time
                set_key(env_path, 'USER_ID', user_id)
                set_key(env_path, 'ACTIVE_CODE', res.data[0]['data']['active_code'])
                set_key(env_path, 'ACTIVE_INFO', f'脚本已激活，到期时间是:{expire_time}，在此期间祝你玩的愉快。')
                os.environ['CHATGPT_BASE_URL'] = res.data[0]['data']['env']['CHATGPT_BASE_URL']
                os.environ['API_URL'] = res.data[0]['data']['env']['API_URL']
            else:
                print("激活异常:----->", res.data[0]['msg'])
                return gr.update(visible=True), gr.update(value=f"激活异常，{res.data[0]['msg']}")
            return gr.update(visible=False), gr.update(value='激活成功，请重启webui后生效')
    except APIError as error:
        return gr.update(visible=True), gr.update(value='激活失败，请查看控制台信息----->{}'.format(error))
    except AuthApiError as err:
        return gr.update(visible=True), gr.update(value='激活失败，请查看控制台信息----->{}'.format(err))
    return gr.update(visible=True), gr.update(value='激活失败，请查看控制台信息，有疑问请联系开发者')


def get_and_deal_azure_speech_list():
    global vop
    old_azure_content = f'{vop.azure}'
    try:
        voice_number = vop.azure['voice_number']
    except KeyError:
        voice_number = 0
    appid = env_data['BAIDU_TRANSLATE_APPID']
    sec_key = env_data['BAIDU_TRANSLATE_KEY']
    chinese_speech_list = []
    headers = {
        'Ocp-Apim-Subscription-Key': env_data['AZURE_SPEECH_KEY']
    }
    azure_speech_list_response = requests.get(vop.azure['speech_list_url'], headers=headers)
    if azure_speech_list_response.status_code == 200:
        speech_list = azure_speech_list_response.json()
        for i, speech in enumerate(speech_list):
            if 'Chinese' in speech['LocaleName']:
                chinese_speech_list.append(speech)
        if voice_number != len(chinese_speech_list):
            vop.azure['emotion_category'] = {}
            vop.azure['voice_number'] = len(chinese_speech_list)
            vop.azure['voice_role'] = []
            vop.azure['voice_code'] = []
            print("微软语音配置写入中，请稍后...")
            for speech in chinese_speech_list:
                if 'StyleList' in speech or 'RolePlayList' in speech:
                    styles_en = []
                    styles_zh = []
                    roles_en = []
                    roles_zh = []
                    if 'StyleList' in speech:
                        styles_en = speech['StyleList']
                        styles_zh = translate(speech['StyleList'], appid, sec_key)
                    if 'RolePlayList' in speech:
                        roles_en = speech['RolePlayList']
                        roles_zh = translate(speech['RolePlayList'], appid, sec_key)
                    name = speech['DisplayName']
                    if name not in vop.azure['emotion_category']:
                        if len(styles_en) > 0 and len(roles_en) == 0:
                            vop.azure['emotion_category'][name] = {
                                'styles_en': styles_en,
                                'styles_zh': styles_zh
                            }
                        if len(styles_en) == 0 and len(roles_en) > 0:
                            vop.azure['emotion_category'][name] = {
                                'roles_en': roles_en,
                                'roles_zh': roles_zh
                            }
                        if len(styles_en) > 0 and len(roles_en) > 0:
                            vop.azure['emotion_category'][name] = {
                                'styles_en': styles_en,
                                'styles_zh': styles_zh,
                                'roles_en': roles_en,
                                'roles_zh': roles_zh
                            }
                    else:
                        if len(styles_en) > 0:
                            vop.azure['emotion_category'][name]['styles_en'] = styles_en
                            vop.azure['emotion_category'][name]['styles_zh'] = styles_zh
                        if len(roles_en) > 0:
                            vop.azure['emotion_category'][name]['roles_en'] = roles_en
                            vop.azure['emotion_category'][name]['roles_zh'] = roles_zh

                vop.azure['voice_role'].append(speech["LocalName"])

                vop.azure['voice_code'].append(speech["ShortName"])
            vop.azure['speech_list_url'] = 'https://eastus.tts.speech.microsoft.com/cognitiveservices/voices/list'
            vop.azure['aue'] = ['mp3', 'wav', 'pcm']
            if sys.platform == 'win32':
                file_path = os.getcwd() + '\\extensions\\batch-draw\\scripts\\voice_params.py'
            else:
                file_path = os.getcwd() + '/extensions/batch-draw/scripts/voice_params.py'
            with open(file_path) as f:
                content = f.read()
            content = content.replace(old_azure_content, f'{vop.azure}')
            with open(file_path, "w") as f:
                f.write(content)
            from scripts import voice_params as vop
            print("微软语音配置写入完成")


role_dict = {
    '知甜_多情感': (vop.ali['emotion_category']['zhitian_emo_zh'], vop.ali['emotion_category']['zhitian_emo_en']),
    '知米_多情感': (vop.ali['emotion_category']['zhimi_emo_zh'], vop.ali['emotion_category']['zhimi_emo_en']),
    '知妙_多情感': (vop.ali['emotion_category']['zhimiao_emo_zh'], vop.ali['emotion_category']['zhimiao_emo_en']),
    '知燕_多情感': (vop.ali['emotion_category']['zhiyan_emo_zh'], vop.ali['emotion_category']['zhiyan_emo_en']),
    '知贝_多情感': (vop.ali['emotion_category']['zhibei_emo_zh'], vop.ali['emotion_category']['zhibei_emo_en'])
}


def cmdargs(line):
    args = shlex.split(line)
    pos = 0
    res = {}

    while pos < len(args):
        arg = args[pos]

        assert arg.startswith("--"), f'must start with "--": {arg}'
        tag = arg[2:]

        func = prompt_tags.get(tag, None)
        assert func, f'unknown commandline option: {arg}'

        assert pos + 1 < len(args), f'missing argument for command line option {arg}'

        val = args[pos + 1]

        res[tag] = func(val)

        pos += 2

    return res


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


# analyzing parameters by imitating mj
def parse_args(args_str, default=None):
    parser = argparse.ArgumentParser()
    arg_dict = {}
    arg_name = None
    arg_values = []
    if args_str != "":
        for arg in args_str.split():
            if arg.startswith("@"):
                if arg_name is not None:
                    if len(arg_values) == 1:
                        arg_dict[arg_name] = arg_values[0]
                    elif len(arg_values) > 1:
                        arg_dict[arg_name] = arg_values[0:]
                    else:
                        arg_dict[arg_name] = default
                arg_name = arg.lstrip("@")
                arg_values = []
            else:
                arg_values.append(arg)

        if arg_name is not None:
            if len(arg_values) == 1:
                arg_dict[arg_name] = arg_values[0]
            elif len(arg_values) > 1:
                arg_dict[arg_name] = arg_values[0:]
            else:
                arg_dict[arg_name] = default

    return arg_dict


def run_pip(args, desc=None):
    index_url = os.environ.get('INDEX_URL', "")
    python = sys.executable
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {args} -i https://pypi.douban.com/simple', desc=f"执行操作: {desc}",
               errdesc=f"Couldn't install {desc}")


def run(command, desc=None, errdesc=None, custom_env=None):
    if desc is not None:
        print(desc)

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
                            env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:
        message = f"""{errdesc or 'Error running command'}.
                        Command: {command}
                        Error code: {result.returncode}
                        stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout) > 0
        else '<empty>'}
                        stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr) > 0
        else '<empty>'}
                    """
        raise RuntimeError(message)

    return result.stdout.decode(encoding="utf8", errors="ignore")


# Counting the number of folders in a folder
def count_subdirectories(path):
    count = 0
    for item in os.listdir(path):
        # 跳过以'.'开头的子目录
        if item.startswith('.'):
            continue

        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            count += 1
    return count


def baidu_translate(query, from_lang, to_lang, appid, secret_key):
    translated_text = ""
    salt = random.randint(32768, 65536)
    sign_str = appid + query + str(salt) + secret_key
    sign = hashlib.md5(sign_str.encode('utf-8')).hexdigest()
    url = f'https://fanyi-api.baidu.com/api/trans/vip/translate?q={query}&from={from_lang}&to={to_lang}' \
          f'&appid={appid}&salt={salt}&sign={sign}'
    resp = requests.get(url)
    result = json.loads(resp.text)
    for i, item in enumerate(result['trans_result']):
        translated_text += item['dst']
        if i != len(result['trans_result']) - 1:
            translated_text += '\n'
    return translated_text


def translate(text_list, appid, key):
    result = []
    for text in text_list:
        result.append(baidu_translate(text, 'en', 'zh', appid, key))
    return result


def refresh_active_data():
    global env_data, realtime, is_expired
    if userid != '':
        res_data = call_rpc_function(userid, active_code, machine_code)
        try:
            if res_data.data[0]['code'] == 0:
                env_data = res_data.data[0]['data']['env']
                set_key(env_path, 'ACTIVE_CODE', res_data.data[0]['data']['active_code'])
                env_expire_time = res_data.data[0]['data']['expire_at']
                realtime = chang_time_zone(env_expire_time)
                set_key(env_path, 'EXPIRE_AT', realtime)
                env_data['EXPIRE_AT'] = realtime
                os.environ['CHATGPT_BASE_URL'] = env_data['CHATGPT_BASE_URL']
                os.environ['API_URL'] = env_data['API_URL']
                is_expired = compare_time(realtime)
                info = f'脚本已激活，到期时间是:{realtime}，在此期间祝你玩的愉快。'
                get_and_deal_azure_speech_list()
                set_key(env_path, 'ACTIVE_INFO', info)
            elif res_data.data[0]['code'] == 1:
                is_expired = True
                info = f'脚本已过期，请联系管理员'
                set_key(env_path, 'ACTIVE_INFO', info)
                print(info)
            elif res_data.data[0]['code'] == 4:
                print(res_data.data[0]['msg'])
        except APIError as e:
            print("获取环境配置失败，请重启webui")
        load_dotenv(dotenv_path=env_path, override=True)


refresh_active_data()


def get_last_subdir(path):
    # 获取目录下的所有子目录（不包括以'.'开头的子目录）并按顺序排序
    sub_dirs = natsorted([d for d in os.listdir(path) if not d.startswith('.') and os.path.isdir(os.path.join(path, d))])
    # 如果存在子目录，返回最后一个子目录的完整路径，否则返回None
    return os.path.join(path, sub_dirs[-1]) if sub_dirs else None


# All the image processing is done in this method
def process(p, prompt_txt, prompts_folder, max_frames, custom_font, text_font_path, text_watermark, text_watermark_color,
            text_watermark_content, text_watermark_font, text_watermark_pos, text_watermark_size, text_watermark_target, save_or,
            default_prompt_type, need_default_prompt, need_negative_prompt, need_combine_prompt, combine_prompt_type, cb_h, cb_w,
            lora_name, batch_images):
    p.n_iter = batch_images
    if prompts_folder == "":
        folder_prefix = os.getcwd() + "/" if sys.platform != 'win32' else os.getcwd() + "\\"
        prompts_folder = folder_prefix + novel_tweets_generator_prompts_folder
        if get_last_subdir(prompts_folder) is not None:
            prompts_folder = get_last_subdir(prompts_folder)
        else:
            print(f"提示词文件夹{prompts_folder}中不存在提示词文件")
            return
        prompts_folder = prompts_folder.replace("\\", "/")
        prompts_folder = prompts_folder.replace('"', '')
    prompts_folders = count_subdirectories(prompts_folder)
    count = 0
    for root, dirs, files in os.walk(prompts_folder):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if not file.startswith('.'):
                count += 1

    frames = []
    results = []
    processed_list = []
    if prompts_folders == 0:
        files_and_dirs = os.listdir(prompts_folder)
        filtered_files_and_dirs = [f for f in files_and_dirs if not f.startswith('.')]
        file_count = len(filtered_files_and_dirs)
        state.job_count = min(int(file_count * p.n_iter), max_frames * 1)
        filenames = []
        result = dura, first_processed, original_images, processed_images, \
            processed_images2, frames_num, cp, cps = deal_with_single_image(max_frames, p, prompt_txt, prompts_folder,
                                                                            default_prompt_type, need_default_prompt,
                                                                            need_negative_prompt, need_combine_prompt,
                                                                            combine_prompt_type, cb_h, cb_w, lora_name, batch_images)
        frames.append(frames_num)
        filenames.append(os.path.basename(prompts_folder))
        images_post_processing(custom_font, filenames, frames, original_images, cp,
                               first_processed, processed_images,
                               processed_images2, save_or, text_font_path, text_watermark,
                               text_watermark_color,
                               text_watermark_content, text_watermark_font, text_watermark_pos,
                               text_watermark_size,
                               text_watermark_target, cps, batch_images)
        results.append(result)
    else:
        state.job_count = min(int(count * p.n_iter), max_frames * prompts_folders)
        for file_name in os.listdir(prompts_folder):
            if file_name.startswith('.'):
                continue
            folder_path = os.path.join(prompts_folder, file_name)
            if os.path.isdir(folder_path):
                filenames = []
                result = dura, first_processed, original_images, processed_images, \
                    processed_images2, frames_num, cp, cps = deal_with_single_image(max_frames, p, prompt_txt, folder_path,
                                                                                    default_prompt_type, need_default_prompt,
                                                                                    need_negative_prompt, need_combine_prompt,
                                                                                    combine_prompt_type, cb_h, cb_w, lora_name, batch_images)
                frames.append(frames_num)
                filenames.append(os.path.basename(folder_path))
                images_post_processing(custom_font, filenames, frames, original_images, cp,
                                       first_processed, processed_images,
                                       processed_images2, save_or, text_font_path, text_watermark,
                                       text_watermark_color,
                                       text_watermark_content, text_watermark_font, text_watermark_pos,
                                       text_watermark_size,
                                       text_watermark_target, cps, batch_images)
                results.append(result)

    for result in results:
        dura = result[0]
        processed_list.append(result[1])

    first_processed = merge_processed_objects(processed_list)
    return first_processed


def merge_processed_objects(processed_list):
    if len(processed_list) == 0:
        return None

    merged_processed = processed_list[0]
    for processed in processed_list[1:]:
        merged_processed.images.extend(processed.images)
        merged_processed.all_prompts.extend(processed.all_prompts)
        merged_processed.all_negative_prompts.extend(processed.all_negative_prompts)
        merged_processed.all_seeds.extend(processed.all_seeds)
        merged_processed.all_subseeds.extend(processed.all_subseeds)
        merged_processed.infotexts.extend(processed.infotexts)

    return merged_processed


def get_file_name_without_extension(file_path):
    file_name = os.path.basename(file_path)
    name_without_extension = os.path.splitext(file_name)[0]
    return name_without_extension


def get_prompts(default_prompt_dict, prompt_keys):
    keys = [str(int(i)) + '.' for i in prompt_keys.split('+')]
    result_str = ''

    for key in keys:
        for dict_key in default_prompt_dict:
            if dict_key.startswith(key):
                result_str += default_prompt_dict[dict_key] + ", "
                break
    result_str = re.sub(r',(?!\s)', ', ', result_str)
    result_str = re.sub(r', , ', ', ', result_str)
    words = re.split(', ', result_str)
    final_words = list(dict.fromkeys(words))
    result_str = ', '.join(final_words)
    tags = re.findall(r'<.*?>, ', result_str)
    result_str = re.sub(r'<.*?>, ', '', result_str)
    result_str += ''.join(tags)
    return result_str


def deal_with_single_image(max_frames, p, prompt_txt, prompts_folder, default_prompt_type, need_default_prompt, need_negative_prompt,
                           need_combine_prompt, combine_prompt_type, cb_h, cb_w, lora_name, batch_images):
    cps = []
    assert os.path.isdir(prompts_folder), f"关键词文件夹-> '{prompts_folder}' 不存在或不是文件夹."
    prompt_files = natsorted([f for f in os.listdir(prompts_folder) if os.path.isfile(os.path.join(prompts_folder, f)) and not f.startswith('.')])
    p.n_iter = batch_images
    original_images = []
    processed_images = []
    processed_images2 = []
    for i in range(p.batch_size * p.n_iter):
        original_images.append([])
        processed_images.append([])
        processed_images2.append([])

    default_prompt_dict = {
        "1.基本提示(通用)": ps.default_prompts,
        "2.基本提示(通用修手)": ps.default_prompts_fix_hands,
        "3.基本提示(增加细节1)": ps.default_prompts_add_details_1,
        "4.基本提示(增加细节2)": ps.default_prompts_add_details_2,
        "5.基本提示(梦幻童话)": ps.default_prompts_fairy_tale
    }

    if not need_default_prompt and default_prompt_type in default_prompt_dict:
        prompt_txt = default_prompt_dict[default_prompt_type]

    if not need_default_prompt and need_combine_prompt:
        prompt_txt = get_prompts(default_prompt_dict, combine_prompt_type)

    lines = [x.strip() for x in prompt_txt.splitlines()]
    lines = [x for x in lines if len(x) > 0]

    p.do_not_save_grid = True

    job_count = 0
    jobs = []

    for line in lines:
        if "--" in line:
            try:
                args = cmdargs(line)
            except Exception as e:
                print(f"Error parsing line [line] as commandline:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                args = {"prompt": line}
        else:
            args = {"prompt": line}

        n_iter = args.get("n_iter", 1)
        job_count += 1
        jobs.append(args)

    j = -1
    file_idx = 0
    frame_count = 0
    copy_p = copy.copy(p)
    if not need_negative_prompt:
        copy_p.negative_prompt = ps.default_negative_prompts

    for n, prompt_file in enumerate(prompt_files):
        if state.interrupted:
            state.nextjob()
            break
        if state.skipped:
            state.skipped = False
        state.job = f"{state.job_no + 1} out of {state.job_count}"
        j = j + 1
        if frame_count >= max_frames:
            break
        for k, v in args.items():
            setattr(copy_p, k, v)
        if file_idx < len(prompt_files):
            prompt_file = os.path.join(prompts_folder, prompt_files[file_idx])

            with open(prompt_file, "rb") as f:
                if is_installed('chardet'):
                    import chardet
                result = chardet.detect(f.read())
                file_encoding = result['encoding']
            with open(prompt_file, "r", encoding=file_encoding) as f:
                individual_prompt = f.read().strip()
            parsed_args = parse_args(individual_prompt)
            index = individual_prompt.find("@")
            if index != -1:
                individual_prompt = individual_prompt[:index]
            if bool(parsed_args):
                only_digits = re.sub('[^0-9]', '', list(parsed_args.keys())[0])
                character_num = int(only_digits)
                individual_prompt = individual_prompt + f'{pc.character_prompts[character_num]}'
            copy_p.prompt = f"{copy_p.prompt}, {individual_prompt}, {lora_name}"
            file_idx += 1
        if cb_h:
            copy_p.width = 576
            copy_p.height = 1024
        elif cb_w:
            copy_p.width = 1024
            copy_p.height = 576
        for k in range(batch_images):
            copy_p.seed = int(random.randrange(4294967294))
            processed = process_images(copy_p)
            cps.append(processed)
        frame_count += 1
    for j in range(len(cps)):
        for i, img1 in enumerate(cps[j].images):
            if i > 0:
                break
            original_images[i].append(img1)
            processed_images[i].append(img1)
            processed_images2[i].append(img1)
    copy_cp = copy.deepcopy(cps)
    final_processed = merge_processed_objects(cps)
    if len(cps) > 1:
        copy_cp.insert(0, process_images(p))
        final_processed = merge_processed_objects(copy_cp)
    return 0, final_processed, original_images, processed_images, processed_images2, frame_count * p.n_iter, copy_p, cps


# Add a user-specified background image to the image
def add_background_image(foreground_path, background_path, p, processed, filename, cp):
    images_dir = f"{novel_tweets_generator_images_folder}/{formatted_date}"
    cwd = os.getcwd()
    foreground_path = cwd + "/" + foreground_path[0]
    foreground_path = foreground_path.replace("\\", "/")
    foreground_path = foreground_path.replace('"', '')
    foreground = Image.open(foreground_path).convert('RGBA')
    background = Image.open(background_path).convert('RGBA')
    background = background.resize(foreground.size)
    combined = Image.alpha_composite(background, foreground)
    with_bg_image = images.save_image(combined, f"{images_dir}/{filename}/add_bg_images", "",
                                      prompt=cp.prompt, seed=cp.seed, grid=False, p=p,
                                      save_to_dirs=False, info=cp.info)
    return with_bg_image


# Add text watermark
def add_watermark(need_add_watermark_images, need_add_watermark_images1, new_images, or_images,
                  text_watermark_color, text_watermark_content, text_watermark_pos, text_watermark_target,
                  text_watermark_size, text_watermark_font, custom_font, text_font_path, p, processed, filenames,
                  frames, cps, batch_images):
    print("帧数是:", frames)
    text_font = 'msyh.ttc'
    if not custom_font:
        if text_watermark_font == '微软雅黑':
            text_font = 'msyh.ttc'
        elif text_watermark_font == '宋体':
            text_font = 'simsun.ttc'
        elif text_watermark_font == '黑体':
            text_font = 'simhei.ttf'
        elif text_watermark_font == '楷体':
            text_font = 'simkai.ttf'
        elif text_watermark_font == '仿宋宋体':
            text_font = 'simfang.ttf'
    else:
        text_font = text_font_path
        text_font = text_font.replace("\\", "/")
        text_font = text_font.replace('"', '')
    fill = tuple(int(text_watermark_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
    tmp_images = []
    tmp_images1 = []
    watered_images = []
    font = ImageFont.truetype(text_font, size=int(text_watermark_size))
    text_width, text_height = font.getsize(text_watermark_content)
    if int(text_watermark_target) == 1:
        need_add_watermark_images = new_images.copy()
    elif int(text_watermark_target) == 0:
        need_add_watermark_images = or_images.copy()
    elif int(text_watermark_target) == 2:
        need_add_watermark_images = new_images.copy()
        need_add_watermark_images1 = or_images.copy()

    watermarked_images = []
    for i in range(len(need_add_watermark_images)):
        watermarked_images.append([])
    for i in range(len(need_add_watermark_images1)):
        watermarked_images.append([])
    pictures_list1 = []
    pictures_list2 = []
    start = 0
    if int(text_watermark_target) == 2:
        for frame in frames:
            end = start + frame
            pictures_list1.append(need_add_watermark_images[start:end])
            pictures_list2.append(need_add_watermark_images1[start:end])
            start = end
    else:
        for frame in frames:
            end = start + frame
            pictures_list1.append(need_add_watermark_images[start:end])
            start = end
    for j, filename in enumerate(filenames):
        # 初始化计数器和场景索引
        counter = 0
        scene_idx = 0
        print('处理的图片数量是:', len(pictures_list1[j]))
        for i, img in enumerate(pictures_list1[j]):
            if int(text_watermark_target) == 0:
                text_overlay_image = Image.new('RGBA', img.size, (0, 0, 0, 0))
            else:
                cwd = os.getcwd()
                bg_path = cwd + "/" + img[0]
                bg_path = bg_path.replace("\\", "/")
                bg_path = bg_path.replace('"', '')
                bg = Image.open(bg_path).convert('RGBA')
                text_overlay_image = Image.new('RGBA', bg.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_overlay_image)
            x = 0
            y = 0
            if int(text_watermark_pos) == 0:
                x = (text_overlay_image.width - text_width) / 2
                y = (text_overlay_image.height - text_height) / 2
            elif int(text_watermark_pos) == 1:
                x = 10
                y = 10
            elif int(text_watermark_pos) == 2:
                x = text_overlay_image.width - text_width - 10
                y = 10
            elif int(text_watermark_pos) == 3:
                x = 10
                y = text_overlay_image.height - text_height - 10
            elif int(text_watermark_pos) == 4:
                x = text_overlay_image.width - text_width - 10
                y = text_overlay_image.height - text_height - 10
            draw.text((x, y), text_watermark_content, font=font, fill=fill)
            if int(text_watermark_target) == 0:
                (fullfn, _) = images.save_image(img,
                                                f"{novel_tweets_generator_images_folder}/{formatted_date}/{filename}/watermarked_images",
                                                "tmp",
                                                prompt=p.prompt_for_display, seed=processed.seed, grid=False,
                                                p=p,
                                                save_to_dirs=False, info=processed.info)
                tmp_images.append(fullfn)
                original_dir, original_filename = os.path.split(fullfn)
                original_image = Image.open(fullfn)
                original_image = original_image.convert("RGBA")
                watermarked_image = Image.alpha_composite(original_image, text_overlay_image)
                original_filename = original_filename.replace("tmp-", "")
            else:
                watermarked_image = Image.alpha_composite(bg.convert('RGBA'), text_overlay_image)
                original_dir, original_filename = os.path.split(img[0])
            # 当计数器达到batch_images时，更新场景索引并重置计数器
            if counter == batch_images:
                scene_idx += 1
                counter = 0
            # 根据场景索引创建文件夹
            images_dir = f"{novel_tweets_generator_images_folder}/{formatted_date}/{filename}/watermarked_images/scene{scene_idx + 1}"
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            watermarked_path = os.path.join(images_dir, f"{original_filename}")
            watermarked_image.save(watermarked_path)
            img1 = Image.open(watermarked_path)
            watered_images.append(img1)
            counter += 1
        # 重置计数器和场景索引
        counter = 0
        scene_idx = 0
        if int(text_watermark_target) == 2:
            for i, img in enumerate(pictures_list2[j]):
                x = 0
                y = 0
                if int(text_watermark_pos) == 0:
                    x = (img.width - text_width) / 2
                    y = (img.height - text_height) / 2
                elif int(text_watermark_pos) == 1:
                    x = 10
                    y = 10
                elif int(text_watermark_pos) == 2:
                    x = img.width - text_width - 10
                    y = 10
                elif int(text_watermark_pos) == 3:
                    x = 10
                    y = img.height - text_height - 10
                elif int(text_watermark_pos) == 4:
                    x = img.width - text_width - 10
                    y = img.height - text_height - 10
                (fullfn, _) = images.save_image(img,
                                                f"{novel_tweets_generator_images_folder}/{formatted_date}/{filename}/watermarked_images",
                                                "tmp",
                                                prompt=p.prompt_for_display, seed=processed.seed, grid=False,
                                                p=p,
                                                save_to_dirs=False, info=processed.info)
                tmp_images1.append(fullfn)
                original_dir, original_filename = os.path.split(fullfn)
                original_image = Image.open(fullfn)
                original_image = original_image.convert("RGBA")
                transparent_layer = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(transparent_layer)
                draw.text((x, y), text_watermark_content, font=font, fill=fill)
                watermarked_image = Image.alpha_composite(original_image, transparent_layer)
                original_filename = original_filename.replace("tmp-", "")
                # 当计数器达到batch_images时，更新场景索引并重置计数器
                if counter == batch_images:
                    scene_idx += 1
                    counter = 0
                # 根据场景索引创建文件夹
                images_dir = f"{novel_tweets_generator_images_folder}/{formatted_date}/{filename}/watermarked_images/scene{scene_idx + 1}"
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)
                watermarked_path = os.path.join(images_dir, f"{original_filename}")
                watermarked_image.save(watermarked_path)
                img1 = Image.open(watermarked_path)
                watered_images.append(img1)
    for path in tmp_images:
        os.remove(path)
    for path in tmp_images1:
        os.remove(path)
    for i, img in enumerate(watered_images):
        watermarked_images[i].append(img)
    return watermarked_images


# Transparent background and add a specified background
def remove_bg(add_bg, bg_path, p, processed, processed_images2, rm_bg, filenames, cps):
    new_images = []
    final_images = []
    images_dir = f"{novel_tweets_generator_images_folder}/{formatted_date}"
    if rm_bg:
        for i, filename in enumerate(filenames):
            for j, img in enumerate(processed_images2[i]):
                new_image = images.save_image(img, f"{images_dir}/{filename}/rm_bg_images", "",
                                              prompt=cps[j].prompt, seed=cps[j].seed, grid=False, p=p,
                                              save_to_dirs=False, info=cps[j].info)
                new_images.append(new_image)
                if add_bg:
                    if bg_path == "":
                        print("未找到要使用的背景图片,将不会添加背景图")
                    else:
                        bg_path = bg_path.replace("\\", "/")
                        bg_path = bg_path.replace('"', '')
                        img1 = add_background_image(new_image, bg_path, p, processed, filename, cps[j])
                        final_images.append(img1)

        if not add_bg:
            final_images = new_images

    return final_images


# Image post-processing
def images_post_processing(custom_font, filenames, frames, original_images, p,
                           processed, processed_images,
                           processed_images2, save_or, text_font_path, text_watermark,
                           text_watermark_color,
                           text_watermark_content, text_watermark_font, text_watermark_pos,
                           text_watermark_size,
                           text_watermark_target, cps, batch_images):
    processed_images_flattened = []
    # here starts the custom image saving logic
    if save_or:
        scene_idx = 0
        counter = 0
        for i, filename in enumerate(filenames):
            for j, img in enumerate(original_images[i]):
                # 当计数器达到batch_images时，更新场景索引并重置计数器
                if counter == batch_images:
                    scene_idx += 1
                    counter = 0
                images_dir = f"{novel_tweets_generator_images_folder}/{formatted_date}/{filename}/original_images/scene{scene_idx + 1}"
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)
                images.save_image(img, images_dir, "",
                                  prompt=cps[j].prompt, seed=cps[j].seed, grid=False, p=p,
                                  save_to_dirs=False, info=cps[j].info)
                counter += 1
        # here is the original picture show
        for row in original_images:
            processed_images_flattened += row
            if len(processed_images_flattened) == 1:
                processed.images = processed_images_flattened
            else:
                processed.images = [images.image_grid(processed_images_flattened,
                                                      rows=p.batch_size * p.n_iter)] + processed_images_flattened
    or_images = []
    if len(processed.images) == 1:
        or_images.append(processed.images[0])
    else:
        for i, img in enumerate(processed.images):
            if i == 0:
                continue
            or_images.append(processed.images[i])
    need_add_watermark_images = []
    need_add_watermark_images1 = []
    new_images = []
    # Operation after adding a text watermark
    if text_watermark:
        watermarked_images = add_watermark(need_add_watermark_images, need_add_watermark_images1, new_images,
                                           or_images, text_watermark_color, text_watermark_content, text_watermark_pos,
                                           text_watermark_target, text_watermark_size, text_watermark_font, custom_font,
                                           text_font_path, p, processed, filenames, frames, cps, batch_images)
        # After adding the watermark, only the final image will be displayed
        processed_images_flattened = []
        for row in watermarked_images:
            processed_images_flattened += row
        if len(processed_images_flattened) == 1:
            processed.images = processed_images_flattened
        else:
            processed.images = [images.image_grid(processed_images_flattened,
                                                  rows=p.batch_size * p.n_iter)] + processed_images_flattened


def ai_process_article(ai_prompt, original_article, scene_number, api_cb, use_proxy):
    if compare_time(env_data['EXPIRE_AT']):
        print("脚本已到期")
        return gr.update(value='脚本已到期'), gr.update(interactive=True)
    proxy = None
    default_pre_prompt = """首先Stable Diffusion是一款利用深度学习的文生图模型，支持通过使用提示词来产生新的图像，描述要包含或省略的元素。 我在这里引入 Stable Diffusion 
    算法中的 Prompt 概念，又被称为提示符。 这里的 Prompt 通常可以用来描述图像，他由普通常见的单词构成，最好是可以在数据集来源站点找到的著名标签（比如 Danbooru)。 
    下面我将说明 Prompt 的生成步骤，这里的 Prompt 主要用于描述人物。 在 Prompt 的生成中，你需要通过提示词来描述 人物属性，主题，外表，情绪，衣服，姿势，视角，动作，背景 。 
    用单词或短语甚至自然语言的标签来描述，并不局限于我给你的单词。 然后将你想要的相似的提示词组合在一起，请使用英文半角 , 做分隔符，并将这些按从最重要到最不重要的顺序 排列。  
    人物属性中，1girl 表示你生成了一个女孩，1boy 表示你生成了一个男孩，人数可以多人。 另外注意，Prompt中不能带有-和_。可以有空格和自然语言，但不要太多，单词不能重复。 
    包含人物性别、主题、外表、情绪、衣服、姿势、视角、动作、背景，将这些按从最重要到最不重要的顺序排列,请尝试生成故事分镜的Prompt,细节越多越好。
    其次你是专业的场景分镜描述专家，我给你一段文字，首先你需要将文字内容改得更加吸引人，然后你需要把修改后的文字分为不同的场景分镜。每个场景必须要细化，要给出人物，时间，地点，
    场景的描述，如果分镜不存在人物就写无人。必须要细化环境描写（天气，周围有些什么等等内容），必须要细化人物描写（人物衣服，衣服样式，衣服颜色，表情，动作，头发，发色等等），
    如果多个分镜中出现的人物是同一个，请统一这个人物的衣服，发色等细节。如果分镜中出现多个人物，还必须要细化每个人物的细节。
    你回答的分镜要加入自己的一些想象，但不能脱离原文太远。你的回答请务必将每个场景的描述转换为单词，并使用多个单词描述场景，每个分镜至少6个单词，如果分镜中出现了人物,请添加人物
    数量的描述。
    你还需要分析场景分镜中各个物体的比重并且将比重按照提示的格式放在每个单词的后面。你只用回复场景分镜内容，其他的不要回复。
    例如这一段话：我和袁绍是大学的时候认识的，在一起了三年。毕业的时候袁绍说带我去他家见他爸妈。去之前袁绍说他爸妈很注重礼节。还说别让我太破费。我懂，我都懂......
    于是我提前去了我表哥顾朝澜的酒庄随手拿了几瓶红酒。临走我妈又让我再带几个LV的包包过去，他妈妈应该会喜欢的。我也没多拿就带了两个包，其中一个还是全球限量版。女人哪有不喜欢包的，
    所以我猜袁绍妈妈应该会很开心吧。
    将它分为四个场景，你可能需要这样回答我：
    1. 情侣, (一个女孩和一个男孩:1.5), (女孩黑色的长发:1.2), 微笑, (白色的裙子:1.2), 非常漂亮的面庞, (女孩手挽着一个男孩:1.5), 男孩黑色的短发, (穿着灰色运动装, 
    帅气的脸庞:1.2), 走在大学校园里, 
    2. 餐馆内, 一个女孩, (黑色的长发, 白色的裙子:1.5), 坐在餐桌前, 一个男孩坐在女孩的对面, (黑色的短发, 灰色的外套:1.5), 两个人聊天, 
    3. 酒庄内, 一个女孩, 微笑, (黑色的长发, 白色的裙子:1.2),(站着:1.5), (拿着1瓶红酒:1.5), 
    4. 一个女孩, (白色的裙子, 黑色的长发:1.5),(手上拿着两个包:1.5), 站在豪华的客厅内, 
    不要拘泥于我给你示例中的权重数字，权重的范围在1到2之前的权重值。你需要按照分镜中的画面自己判断权重。注意回复中的所有标点符号请使用英文的标点符号包括逗号，不要出现句号，
    仿照例子，给出一套详细描述以下内容的prompt。直接开始给出prompt不需要用自然语言描述：请你牢记这些规则，任何时候都不要忘记。
"""
    if ai_prompt != '':
        default_pre_prompt = ai_prompt
    if int(scene_number) != 0:
        prompt = default_pre_prompt + "\n" + f"内容是：{original_article}\n必须将其转换为{int(scene_number)}个场景分镜。"
    else:
        prompt = default_pre_prompt + "\n" + f"内容是：{original_article}\n你需要根据文字内容自己分析可以转换成几个场景分镜。" \
                                             f"你不需要向我解释你转换场景个数和权重的原因，你只用回复场景分镜内容，其他的不要回复"
    response = ""
    if use_proxy:
        proxy = os.environ.get('PROXY')
    if api_cb:
        try:
            openai_key = env_data['KEY']
            chatbot = ChatbotV3(api_key=openai_key, proxy=proxy if (proxy != "" or proxy is not None) else None, engine='gpt-3.5-turbo-0613',
                                temperature=0.8)
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
    response = response.replace('，', ', ')
    return gr.update(value=response), gr.update(interactive=True)


def deepl_translate_text(api_key, text, target_lang: str = 'EN-US'):
    deepl_url = 'https://api-free.deepl.com/v2/translate'
    params = {
        'auth_key': api_key,
        'text': text,
        'target_lang': target_lang
    }
    response = requests.post(deepl_url, data=params)
    if response.status_code == 200:
        result = json.loads(response.content.decode('utf-8'))
        translated_text = result['translations'][0]['text']
        return translated_text
    else:
        return None


def save_prompts(prompts, is_translate=False):
    if compare_time(env_data['EXPIRE_AT']):
        print("脚本已到期")
        return gr.update(interactive=True)
    if prompts != "":
        prompts = re.sub(r'\n\s*\n', '\n', prompts)
        print("开始处理并保存AI推文")
        appid = env_data['BAIDU_TRANSLATE_APPID']
        key = env_data['BAIDU_TRANSLATE_KEY']
        deepl_api_key = env_data['DEEPL_API_KEY']
        if is_translate:
            prompts = deepl_translate_text(deepl_api_key, prompts)
            if prompts is None:
                prompts = baidu_translate(prompts, 'zh', 'en', appid, key)
        current_folders = count_subdirectories(novel_tweets_generator_prompts_folder)
        novel_tweets_generator_prompts_sub_folder = 'outputs/novel_tweets_generator/prompts/' + f'{current_folders + 1}'
        os.makedirs(novel_tweets_generator_prompts_sub_folder, exist_ok=True)
        if prompts is not None:
            lines = prompts.splitlines()
            for i, line in enumerate(lines):
                parts = line.split()
                content = ' '.join(parts[1:])
                filename = novel_tweets_generator_prompts_sub_folder + '/scene' + f'{i + 1}.txt'
                content = content.replace(': ', ':').lower()
                with open(filename, 'w') as f:
                    f.write(content)
            full_path = f'{os.getcwd()}/{novel_tweets_generator_prompts_sub_folder}'
            full_path = full_path.replace('\\', '/')
            print("AI推文保存完成，保存路径在:", f'{full_path}  的文件夹内')
            return gr.update(interactive=True)
        else:
            print("AI推文翻译失败")
            return gr.update(interactive=True)
    else:
        print("AI处理的推文为空，不做处理")
        return gr.update(interactive=True)


def change_state(is_checked):
    if is_checked:
        return gr.update(value=False)
    else:
        return gr.update(value=True)


def change_selected(is_checked):
    if is_checked:
        return gr.update(value=False), gr.update(value=False)
    else:
        return gr.update(value=True), gr.update(value=False)


def set_un_clickable():
    return gr.update(interactive=False)


def tts_fun(text, spd, pit, vol, per, aue, tts_type, voice_emotion, voice_emotion_intensity, role_play):
    if compare_time(env_data['EXPIRE_AT']):
        print("脚本已到期")
        return gr.update(interactive=True)
    print("语音引擎类型是:", tts_type)
    if tts_type == "百度":
        tts_baidu(aue, per, pit, spd, text, vol)
    elif tts_type == "阿里":
        tts_ali(text, spd, pit, vol, per, aue, voice_emotion, voice_emotion_intensity)
    elif tts_type == "华为":
        tts_huawei(aue, per, pit, spd, text, vol)
    elif tts_type == '微软':
        tts_azure(text, spd, pit, vol, per, aue, voice_emotion, voice_emotion_intensity, role_play)
    return gr.update(interactive=True)


def tts_baidu(aue, per, pit, spd, text, vol):
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
    # get access token
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
                "cuid": machine_code,
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


def tts_ali(text, spd, pit, vol, per, aue, voice_emotion, voice_emotion_intensity):
    client = AcsClient(
        env_data['ALIYUN_ACCESSKEY_ID'],
        env_data['ALIYUN_ACCESSKEY_SECRET'],
        "cn-shanghai"
    )
    file_count = 0
    for root, dirs, files in os.walk(novel_tweets_generator_audio_folder):
        file_count += len(files)
    token = ''
    is_short = True
    if len(text) > 100:
        is_short = False
    if is_short:
        tts_url = vop.ali['short_voice_url']
    else:
        tts_url = vop.ali['long_voice_url']
    app_key = env_data['ALIYUN_APPKEY']
    for i, role in enumerate(vop.ali['voice_role']):
        if per == role:
            if '多情感' in role:
                for key in role_dict:
                    if key in role:
                        emo_zh, emo_en = role_dict[key]
                        for j, emo in enumerate(emo_zh):
                            if emo == voice_emotion:
                                emo_type = emo_en[j]
                                text = f'<speak><emotion category="{emo_type}" intensity="{voice_emotion_intensity}">' + text + '</emotion></speak>'
            per = vop.ali['voice_code'][i]
    request = CommonRequest()
    request.set_method('POST')
    request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
    request.set_version('2019-02-28')
    request.set_action_name('CreateToken')
    try:
        response = client.do_action_with_exception(request)
        jss = json.loads(response)
        if 'Token' in jss and 'Id' in jss['Token']:
            token = jss['Token']['Id']
    except Exception as e:
        print(e)

    headers = {
        'Content-Type': 'application/json',
        'Accept': '*/*'
    }
    short_payload = json.dumps({
        'text': text,
        'appkey': app_key,
        'token': token,
        'format': aue,
        'voice': per,
        'volume': vol,
        'speech_rate': spd,
        'pitch_rate': pit
    })
    long_payload = json.dumps({
        "payload": {
            "tts_request": {
                "voice": per,
                "sample_rate": 16000,
                "format": aue,
                "text": text,
                "enable_subtitle": False,
                'volume': vol,
                'speech_rate': spd,
                'pitch_rate': pit
            },
            "enable_notify": False
        },
        "context": {
            "device_id": machine_code
        },
        "header": {
            "appkey": app_key,
            "token": token
        }
    })
    if token == "":
        print("阿里云授权失败，请稍后重试")
    else:
        data = short_payload.encode('utf-8') if is_short else long_payload.encode('utf-8')
        response = requests.post(tts_url, headers=headers, data=data)
        if is_short:
            print("阿里短文本转语音任务创建成功，任务完成后自动下载，你可以在此期间做其他的事情。")
            if response.status_code == 200:
                content_type = response.headers['Content-Type']
                if content_type == 'audio/mpeg':
                    file_ext = aue
                    file_path = os.path.join(novel_tweets_generator_audio_folder, f'{file_count + 1}.{file_ext}')
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    if sys.platform == 'win32':
                        print("语音下载完成，保存路径是:----->", os.getcwd() + "\\" + file_path)
                    else:
                        print("语音下载完成，保存路径是:----->", os.getcwd() + "/" + file_path)
                elif content_type == 'application/json':
                    print("语音合成失败，错误原因是:----->", f"{response.json()['message']}")
            else:
                print("语音合成失败，错误原因是:----->", f"{response.json()['message']}")
        else:
            if response.status_code == 200:
                print("阿里长文本转语音任务创建成功，任务完成后自动下载，你可以在此期间做其他的事情。")
                task_id = response.json()['data']['task_id']
                while True:
                    data = {
                        'appkey': app_key,
                        'token': token,
                        'task_id': task_id
                    }
                    response1 = requests.get(tts_url, headers=headers, params=data)
                    rj = response1.json()
                    if response1.status_code == 200:
                        if rj["data"]["audio_address"] is not None:
                            speech_url = rj["data"]["audio_address"]
                            response2 = requests.get(speech_url)
                            file_ext = aue
                            file_path = os.path.join(novel_tweets_generator_audio_folder, f'{file_count + 1}.{file_ext}')
                            with open(file_path, 'wb') as f:
                                f.write(response2.content)
                            if sys.platform == 'win32':
                                print("语音下载完成，保存路径是:----->", os.getcwd() + "\\" + file_path)
                            else:
                                print("语音下载完成，保存路径是:----->", os.getcwd() + "/" + file_path)
                            break
                        else:
                            time.sleep(10)
                    else:
                        print("阿里长文本合成语音失败，原因是----->", f"{rj['error_message']}")
                        break
            else:
                print("阿里长文本转语音任务创建失败，原因是:----->", f"{response.json()['message']}")


def tts_huawei(aue, per, pit, spd, text, vol):
    if len(text) > 100:
        print("文本过长")
    else:
        file_count = 0
        for root, dirs, files in os.walk(novel_tweets_generator_audio_folder):
            file_count += len(files)
        token = ''
        get_access_token_url = vop.huawei['get_access_token_url']
        tts_url = vop.huawei['tts_url']
        get_access_token_payload = json.dumps({
            "auth": {
                "identity": {
                    "methods": ["hw_ak_sk"],
                    "hw_ak_sk": {
                        "access": {
                            "key": env_data['HUAWEI_AK']
                        },
                        "secret": {
                            "key": env_data['HUAWEI_SK']
                        }
                    }
                },
                "scope": {
                    "project": {
                        "name": "cn-east-3"
                    }
                }
            }
        })
        token_headers = {
            'Content-Type': 'application/json'
        }
        for i, role in enumerate(vop.huawei['voice_role']):
            if per == role:
                per = vop.huawei['voice_code'][i]
        response = requests.request("POST", get_access_token_url, headers=token_headers, data=get_access_token_payload)
        token = response.headers["X-Subject-Token"]
        if token != '':
            print("华为鉴权成功")
            tts_headers = {
                'X-Auth-Token': f"{token}",
                'Content-Type': 'application/json'
            }
            tts_payload = {
                'text': text,
                'config': {
                    'audio_format': aue,
                    'sample_rate': '16000',
                    'property': per,
                    'speed': spd,
                    'pitch': pit,
                    'volume': vol
                }
            }
            response_tts = requests.post(tts_url, headers=tts_headers, data=json.dumps(tts_payload))
            rj = response_tts.json()
            if response_tts.status_code == 200:
                print("华为语音合成完成")
                data = rj['result']['data']
                audio_data = base64.b64decode(data)
                file_ext = aue
                file_path = os.path.join(novel_tweets_generator_audio_folder, f'{file_count + 1}.{file_ext}')
                if file_ext == 'mp3':
                    with open(file_path, 'wb') as f:
                        f.write(audio_data)
                else:
                    with wave.open(file_path, 'wb') as f:
                        f.setnchannels(1)
                        f.setsampwidth(2)
                        f.setframerate(16000)
                        f.writeframes(audio_data)
                if sys.platform == 'win32':
                    print("语音下载完成，保存路径是:----->", os.getcwd() + "\\" + file_path)
                else:
                    print("语音下载完成，保存路径是:----->", os.getcwd() + "/" + file_path)
            else:
                print("错误返回值是:", f"{response_tts.json()}")
                print("华为语音合成失败，原因是----->", rj['error_msg'])
        else:
            print("华为鉴权失败,请重试")


def tts_azure(text, spd, pit, vol, per, aue, voice_emotion, voice_emotion_intensity, voice_role_play):
    pits = ['x-low', 'low', 'default', 'medium', 'high', 'x-high']
    real_pit = pits[int(pit) - 1]
    print("微软文本转语音任务创建成功，任务完成后自动下载，你可以在此期间做其他的事情。")
    file_count = 0
    for root, dirs, files in os.walk(novel_tweets_generator_audio_folder):
        file_count += len(files)
    speech_key = env_data['AZURE_SPEECH_KEY']
    service_region = "eastus"
    file_ext = aue
    file_path = os.path.join(novel_tweets_generator_audio_folder, f'{file_count + 1}.{file_ext}')
    if sys.platform == 'win32':
        file_path = file_path.replace("/", "\\")
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=file_path)
    voice_name = ''
    speak_tag = 'version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="zh-CN"'
    for i, role_name in enumerate(vop.azure['voice_role']):
        if per == role_name:
            voice_name = vop.azure['voice_code'][i]
    voice_style = ''
    voice_role = ''
    if '多情感' in per:
        for key, persona in vop.azure['emotion_category'].items():
            if key in voice_name:
                for i, style in enumerate(vop.azure['emotion_category'][key]['styles_zh']):
                    if voice_emotion == style:
                        voice_style = vop.azure['emotion_category'][key]['styles_en'][i]
                if 'roles_en' in vop.azure['emotion_category'][key]:
                    for i, role in enumerate(vop.azure['emotion_category'][key]['roles_zh']):
                        if voice_role_play == role:
                            voice_role = vop.azure['emotion_category'][key]['roles_en'][i]
        if voice_style == 'default':
            style_tag = ''
        else:
            style_tag = f'style="{voice_style}" styledegree="{voice_emotion_intensity}"'
        if voice_role == 'default':
            role_tag = ''
        else:
            role_tag = f'role="{voice_role}"'
        if style_tag != '' or role_tag != '':
            express_as_tag = f'<mstts:express-as {role_tag} {style_tag}><prosody volume="{vol}" pitch="{real_pit}" rate="{spd}">{text}</prosody>' \
                             f'</mstts:express-as>'
        else:
            express_as_tag = text
        text = f'<speak {speak_tag}><voice name="{voice_name}">{express_as_tag}</voice></speak>'
    else:
        text = f'<speak {speak_tag}><voice name="{voice_name}"><prosody volume="{vol}" pitch="{real_pit}" rate="{spd}">{text}</prosody></voice>' \
               f'</speak>'
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_synthesizer.speak_ssml_async(text).get()
    # Check result
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        if sys.platform == 'win32':
            print("语音下载完成，保存路径是:----->", os.getcwd() + "\\" + file_path.replace('/', '\\'))
        else:
            print("语音下载完成，保存路径是:----->", os.getcwd() + "/" + file_path)
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("微软语音合成取消: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("微软语音合成异常: {}".format(cancellation_details.error_details))


def change_tts(tts_type):
    # voice_role, audition, voice_speed, voice_pit, voice_vol, output_type, voice_emotion, voice_emotion_intensity
    if tts_type == '百度':
        return gr.update(choices=vop.baidu['voice_role'], value=vop.baidu['voice_role'][0]), gr.update(visible=True), \
            gr.update(minimum=0, maximum=9, value=5, step=1), gr.update(minimum=0, maximum=9, value=5, step=1), \
            gr.update(minimum=0, maximum=15, value=5, step=1), gr.update(choices=vop.baidu['aue'], value=vop.baidu['aue'][0]), \
            gr.update(visible=False), gr.update(visible=False)
    elif tts_type == '阿里':
        return gr.update(choices=vop.ali['voice_role'], value=vop.ali['voice_role'][0]), gr.update(visible=False), \
            gr.update(minimum=-500, maximum=500, value=0, step=100), gr.update(minimum=-500, maximum=500, value=0, step=100), \
            gr.update(minimum=0, maximum=100, value=50, step=10), gr.update(choices=vop.ali['aue'], value=vop.ali['aue'][0]), \
            gr.update(visible=True), gr.update(visible=True)
    elif tts_type == '华为':
        return gr.update(choices=vop.huawei['voice_role'], value=vop.huawei['voice_role'][0]), gr.update(visible=False), \
            gr.update(minimum=-500, maximum=500, value=0, step=100), gr.update(minimum=-500, maximum=500, value=0, step=100), \
            gr.update(minimum=0, maximum=100, value=50, step=10), gr.update(choices=vop.huawei['aue'], value=vop.huawei['aue'][0]), \
            gr.update(visible=False), gr.update(visible=False)
    elif tts_type == '微软':
        return gr.update(choices=vop.azure['voice_role'], value=vop.azure['voice_role'][0]), gr.update(visible=False), \
            gr.update(minimum=0.5, maximum=2, value=1, step=0.1), gr.update(minimum=1, maximum=6, value=3, step=1), \
            gr.update(minimum=0, maximum=100, value=100, step=5), gr.update(choices=vop.azure['aue'], value=vop.azure['aue'][0]), \
            gr.update(visible=False), gr.update(visible=False)


def change_voice_role(tts_type, role):
    if tts_type == '阿里':
        emo_zh = []
        if '多情感' in role:
            for key in role_dict:
                if key in role:
                    emo_zh, emo_en = role_dict[key]
                    break
            return gr.update(choices=emo_zh, value=emo_zh[0], visible=True), gr.update(minimum=0.01, maximum=2.0, value=1, visible=True), gr.update(
                visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    elif tts_type == '百度':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    elif tts_type == '华为':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    elif tts_type == '微软':
        role_code = ''
        for i, role_name in enumerate(vop.azure['voice_role']):
            if role_name == role:
                role_code = vop.azure['voice_code'][i]
        if '多情感' in role:
            for key, persona in vop.azure['emotion_category'].items():
                if key in role_code:
                    if 'roles_en' in vop.azure['emotion_category'][key]:
                        return gr.update(choices=vop.azure['emotion_category'][key]['styles_zh'],
                                         value=vop.azure['emotion_category'][key]['styles_zh'][0], visible=True), \
                            gr.update(minimum=0.01, maximum=2.0, value=1, visible=True), \
                            gr.update(choices=vop.azure['emotion_category'][key]['roles_zh'],
                                      value=vop.azure['emotion_category'][key]['roles_zh'][0], visible=True)
                    else:
                        return gr.update(choices=vop.azure['emotion_category'][key]['styles_zh'],
                                         value=vop.azure['emotion_category'][key]['styles_zh'][0], visible=True), \
                            gr.update(minimum=0.01, maximum=2.0, value=1, visible=True), \
                            gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


class Script(scripts.Script):

    def title(self):
        return "小说推文图片生成器"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        gr.HTML('<br><a href="{}"><font color=blue>！！！点击这里了解如何使用该脚本！！！</font></a>'.format(instructions))
        with gr.Accordion(label="注册和激活", open=True):
            active_code_input = gr.Textbox(label='激活码', placeholder='请在这里输入激活码，激活后该行会隐藏，首次留空激活会给六小时的免费试用',
                                           visible=is_expired)
            if userid != '' and active_code != '':
                value = '重新激活'
            else:
                value = '注册并激活'
            ensure_sign_up = gr.Button(value=value, visible=is_expired)
            if is_expired:
                active_info_text = '脚本未激活或者已过期，请激活或刷新激活信息！'
            else:
                active_info_text = os.environ.get('ACTIVE_INFO')
            active_info = gr.Textbox(label="激活信息是", value=active_info_text,
                                     interactive=False)
            refresh_active_info = gr.Button(value='刷新激活信息')

            def update_active_info():
                refresh_active_data()
                expired = compare_time(env_data['EXPIRE_AT'])
                if expired:
                    refresh_active_info_text = '脚本未激活或者已过期，请激活或刷新激活信息！'
                else:
                    refresh_active_info_text = os.environ.get('ACTIVE_INFO')
                return gr.update(value=refresh_active_info_text), gr.update(visible=expired), gr.update(visible=expired, value='重新激活')

            refresh_active_info.click(update_active_info, outputs=[active_info, active_code_input, ensure_sign_up])
            ensure_sign_up.click(sign_up, inputs=[active_code_input], outputs=[active_info])
        with gr.Accordion(label="基础属性，必填项，每一项都不能为空", open=True):
            with gr.Column(variant='panel'):
                with gr.Accordion(label="1. 默认提示词相关", open=True):
                    default_prompt_type = gr.Dropdown(
                        [
                            "1.基本提示(通用)",
                            "2.基本提示(通用修手)",
                            "3.基本提示(增加细节1)",
                            "4.基本提示(增加细节2)",
                            "5.基本提示(梦幻童话)"
                        ],
                        label="默认正面提示词类别",
                        value="1.基本提示(通用)")
                    need_combine_prompt = gr.Checkbox(label="需要组合技(组合上方类别)？", value=False)
                    combine_prompt_type = gr.Textbox(
                        label="请输入你需要组合的类别组合，例如2+3，不要组合过多种类", visible=False)

                    def is_show_combine(is_show):
                        return gr.update(visible=is_show)

                    need_combine_prompt.change(is_show_combine, inputs=[need_combine_prompt],
                                               outputs=[combine_prompt_type])

                    with gr.Row():
                        need_default_prompt = gr.Checkbox(label="自行输入默认正面提示词(勾选后上面选择将失效)", value=False)
                        need_negative_prompt = gr.Checkbox(label="自行输入默认负面提示词(勾选后需要自行输入负面提示词)", value=False)

                    prompt_txt = gr.Textbox(label="默认提示词，将影响各帧", lines=3, max_lines=5, value="", visible=False)

                    def is_need_default_prompt(is_need):
                        return gr.update(visible=is_need)

                    need_default_prompt.change(is_need_default_prompt, inputs=[need_default_prompt],
                                               outputs=[prompt_txt])

                    need_mix_models = gr.Checkbox(label="分片使用模型(勾选后请在下方输入模型全名包括后缀,使用 ',' 分隔)", value=False, visible=False)
                    models_txt = gr.Textbox(label="模型集合", lines=3, max_lines=5, value="", visible=False)

                    def is_need_mix_models(is_need):
                        return gr.update(visible=is_need)

                    need_mix_models.change(is_need_mix_models, inputs=[need_mix_models],
                                           outputs=[models_txt])
                    lora_name = gr.Textbox(label="使用Lora（此处输入的Lora将影响所有图片）", lines=1, max_lines=2, value="",
                                           placeholder='由于二次元的Lora众多，风格各异，小说的类型也很多，所有脚本并没有内置Lora，想要出现更好的效果可以在此输入Lora，'
                                                       '格式是<lora:lora的名字:lora的权重>,支持多个lora，例如 <lora:fashionGirl_v54:0.5>, '
                                                       '<lora:cuteGirlMix4_v10:0.6>')
                max_frames = gr.Number(
                    label="2. 输入指定的测试图片数量 (运行到指定数量后停止(不会大于所有文本的总数)，用于用户测试生成图片的效果，最小1帧,测试完成后请输入一个很大的数保证能把所有文本操作完毕)",
                    value=666,
                    min=1
                )
                gr.HTML("3. AI处理内容")
                original_article = gr.Textbox(
                    label="输入推文原文",
                    lines=1,
                    max_lines=6,
                    value=""
                )
                with gr.Accordion(label="3.1 AI处理原文"):
                    ai_prompt = gr.Textbox(label='自行输入AI提示词，一般不需要输入', value='')
                    scene_number = gr.Number(
                        label="输入要生成的场景数量(若改为0则由AI自动推断文本可转换的场景数量)",
                        value=10,
                        min=0
                    )
                    with gr.Row():
                        api_cb = gr.Checkbox(
                            label="使用api方式处理",
                            value=True
                        )
                        web_cb = gr.Checkbox(
                            label="使用web方式处理"
                        )
                        cb_use_proxy = gr.Checkbox(
                            label="使用代理(一般无需代理)"
                        )
                        cb_trans_prompt = gr.Checkbox(
                            label="翻译AI推文,保存推文时生效",
                            value=True
                        )
                        api_cb.change(change_state, inputs=[api_cb], outputs=[web_cb])
                        web_cb.change(change_state, inputs=[web_cb], outputs=[api_cb])
                    preset_character = gr.Dropdown(
                        pc.character_list,
                        label="场景人物预设(仅做展示，自带的预设都没有加入Lora控制，请阅读使用说明书了解如何使用)",
                        value="0.无")

                    with gr.Row():
                        with gr.Column(scale=2, min_width=20):
                            custom_preset_title = gr.Textbox(placeholder='自定义预设标题', label='预设标题', visible=False)
                        with gr.Column(scale=8):
                            custom_preset = gr.Textbox(placeholder='自定义预设内容', label='预设内容', visible=False)
                    add_preset = gr.Button(value='保存', visible=False)

                    def change_character(character):
                        if character == '自定义':
                            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
                        else:
                            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

                    preset_character.change(change_character, inputs=[preset_character], outputs=[custom_preset_title, custom_preset, add_preset])

                    def add_character_preset(preset_title, preset_content):
                        length = len(pc.character_list)
                        preset_title = f"{length - 1}.{preset_title}"
                        pc.character_list.insert(-1, preset_title)
                        pc.character_prompts.append(preset_content)
                        current_file_path = os.path.abspath(__file__)
                        current_dir_path = os.path.dirname(current_file_path)
                        file_path = os.path.join(current_dir_path, 'preset_character.py')
                        with open(file_path, 'w') as f:
                            f.write(f"character_list = {pc.character_list}\n")
                            f.write(f"character_prompts = {pc.character_prompts}\n")
                        return gr.update(choices=pc.character_list, value=pc.character_list[length - 1])

                    add_preset.click(add_character_preset, inputs=[custom_preset_title, custom_preset], outputs=[preset_character])
                    ai_article = gr.Textbox(
                        label="AI处理的推文将显示在这里（建议在每个场景描述后面加入Lora，图片会更好看，人物会更稳定，建议在预设里面加入Lora）",
                        lines=1,
                        max_lines=6,
                        value=""
                    )
                    with gr.Row():
                        deal_with_ai = gr.Button(value="AI处理推文")
                        btn_save_ai_prompts = gr.Button(value="保存AI推文")
                    tb_save_ai_prompts_folder_path = gr.Textbox(
                        label="请输入推文保存的路径，若为空则保存在outputs/novel_tweets_generator/prompts路径下的按序号递增的文件夹下")

                    deal_with_ai.click(set_un_clickable, outputs=[deal_with_ai])
                    deal_with_ai.click(ai_process_article, inputs=[ai_prompt, original_article, scene_number, api_cb, cb_use_proxy],
                                       outputs=[ai_article, deal_with_ai])
                    btn_save_ai_prompts.click(set_un_clickable, outputs=[btn_save_ai_prompts])
                    btn_save_ai_prompts.click(save_prompts, inputs=[ai_article, cb_trans_prompt], outputs=[btn_save_ai_prompts])
                with gr.Accordion(label="3.2 原文转语音"):
                    voice_radio = gr.Radio(['百度', '阿里', '华为', '微软'], label="3.2.1 语音引擎", info='请选择一个语音合成引擎，默认为百度',
                                           value='百度')
                    gr.HTML(value="3.2.2 语音引擎参数")
                    with gr.Row():
                        with gr.Column(scale=15):
                            voice_role = gr.Dropdown(vop.baidu['voice_role'], label="语音角色",
                                                     value=vop.baidu['voice_role'][0])
                        with gr.Column(scale=1, min_width=20):
                            audition = gr.HTML('<br><br><a href="{}"><font color=blue>试听</font></a>'.format(vop.baidu['audition_url']))
                    with gr.Row():
                        voice_emotion = gr.Dropdown(vop.ali['emotion_category']['zhitian_emo_zh'], label="情感类型",
                                                    value=vop.ali['emotion_category']['zhitian_emo_zh'][0], visible=False)
                        voice_emotion_intensity = gr.Slider(0.01, 2.0, label='emo_ints(情感强度)', value=1, visible=False)
                        role_play = gr.Dropdown([], label='角色扮演', value='', visible=False)
                    voice_role.change(change_voice_role, inputs=[voice_radio, voice_role],
                                      outputs=[voice_emotion, voice_emotion_intensity, role_play])
                    with gr.Row():
                        voice_speed = gr.Slider(0, 9, label='Speed(语速)', value=5, step=1)
                        voice_pit = gr.Slider(0, 9, label='Pit(语调)', value=5, step=1)
                        voice_vol = gr.Slider(0, 15, label='Vol(音量)', value=5, step=1)
                    output_type = gr.Dropdown(vop.baidu['aue'], label="输出文件格式", value=vop.baidu['aue'][0])
                    voice_radio.change(change_tts, inputs=[voice_radio],
                                       outputs=[voice_role, audition, voice_speed, voice_pit, voice_vol, output_type, voice_emotion,
                                                voice_emotion_intensity])
                    btn_txt_to_voice = gr.Button(value="原文转语音")
                    btn_txt_to_voice.click(set_un_clickable, outputs=[btn_txt_to_voice])
                    btn_txt_to_voice.click(tts_fun,
                                           inputs=[original_article, voice_speed, voice_pit, voice_vol, voice_role, output_type, voice_radio,
                                                   voice_emotion, voice_emotion_intensity, role_play],
                                           outputs=[btn_txt_to_voice])
                prompts_folder = gr.Textbox(
                    label="4. 输入包含提示词文本文件的文件夹路径",
                    info="默认值为空时处理outputs/novel_tweets_generator/prompts文件夹下的最后一个文件夹",
                    lines=1,
                    max_lines=2,
                    value=""
                )
                batch_images = gr.Number(
                    label="5. 每个画面生成图片的数量",
                    value=1,
                    min=1
                )
                gr.HTML("6. 生成图片类型")
                with gr.Row():
                    cb_h = gr.Checkbox(label='竖图', value=True, info="尺寸为576*1024，即9:16的比例")
                    cb_w = gr.Checkbox(label='横图', info="尺寸为1024*576，即16:9的比例")
                    cb_custom = gr.Checkbox(label='自定义尺寸', info="勾选后请在sd的图片尺寸位置自行输入尺寸")
                    cb_h.select(change_selected, inputs=[cb_h], outputs=[cb_w, cb_custom])
                    cb_w.select(change_selected, inputs=[cb_w], outputs=[cb_h, cb_custom])
                    cb_custom.select(change_selected, inputs=[cb_custom], outputs=[cb_h, cb_w])
                gr.HTML("")

            with gr.Accordion(label="去除背景和保留原图(至少选择一项否则文件夹中没有保留生成的图片)", open=True, visible=False):
                with gr.Column(variant='panel', visible=False):
                    with gr.Row():
                        save_or = gr.Checkbox(label="8. 是否保留原图",
                                              info="为了不影响查看原图，默认选中会保存未删除背景的图片", value=True)

            with gr.Accordion(label="更多操作(打开看看说不定有你想要的功能)", open=False):
                with gr.Column(variant='panel'):
                    with gr.Column():
                        text_watermark = gr.Checkbox(label="7. 添加文字水印", info="自定义文字水印")
                        with gr.Row():
                            with gr.Column(scale=8):
                                with gr.Row():
                                    text_watermark_font = gr.Dropdown(
                                        ["微软雅黑", "宋体", "黑体", "楷体", "仿宋宋体"],
                                        label="内置5种字体,启用自定义后这里失效",
                                        value="微软雅黑")
                                    text_watermark_target = gr.Dropdown(["0", "1", "2"],
                                                                        label="水印添加对象(0:原始,1:透明,2:全部)",
                                                                        value="0", visible=False)
                                    text_watermark_pos = gr.Dropdown(["0", "1", "2", "3", "4"],
                                                                     label="位置(0:居中,1:左上,2:右上,3:左下,4:右下)",
                                                                     value="0")
                            with gr.Column(scale=1):
                                text_watermark_color = gr.ColorPicker(label="自定义水印颜色")
                    with gr.Row():
                        with gr.Column(scale=1):
                            text_watermark_size = gr.Number(label="水印字体大小", value=30, min=30)
                        with gr.Column(scale=7):
                            text_watermark_content = gr.Textbox(label="文字水印内容（不要设置过长的文字会遮挡图片）",
                                                                lines=1,
                                                                max_lines=1,
                                                                value="")
                    with gr.Row():
                        with gr.Column(scale=1):
                            custom_font = gr.Checkbox(label="启用自定义水印字体", info="自定义水印字体")
                        with gr.Column(scale=7):
                            text_font_path = gr.Textbox(
                                label="输入自定义水印字体路径(勾选左边自定义水印字体单选框以及功能5起效)",
                                lines=1, max_lines=2,
                                value=""
                            )
            gr.HTML(f"此设备机器码是：{machine_code}")

        return [active_code_input, ensure_sign_up, active_info, prompt_txt, max_frames, prompts_folder, save_or,
                text_watermark, text_watermark_font, text_watermark_target,
                text_watermark_pos, text_watermark_color, text_watermark_size, text_watermark_content, custom_font, text_font_path,
                default_prompt_type, need_default_prompt, need_negative_prompt, need_combine_prompt, combine_prompt_type, original_article,
                ai_prompt, scene_number, deal_with_ai, ai_article, preset_character, api_cb, web_cb, btn_save_ai_prompts,
                tb_save_ai_prompts_folder_path, cb_use_proxy, cb_trans_prompt, cb_w, cb_h, cb_custom, voice_radio, voice_role, voice_speed, voice_pit,
                voice_vol, audition, btn_txt_to_voice, output_type, voice_emotion, voice_emotion_intensity, lora_name, batch_images,
                custom_preset_title, custom_preset, add_preset]

    def run(self, p, active_code, ensure_sign_up, active_info, prompt_txt, max_frames, prompts_folder, save_or,
            text_watermark, text_watermark_font, text_watermark_target,
            text_watermark_pos, text_watermark_color, text_watermark_size, text_watermark_content, custom_font, text_font_path, default_prompt_type,
            need_default_prompt, need_negative_prompt, need_combine_prompt, combine_prompt_type,
            original_article, ai_prompt, scene_number, deal_with_ai, ai_article, preset_character, api_cb, web_cb, btn_save_ai_prompts,
            tb_save_ai_prompts_folder_path,
            cb_use_proxy, cb_trans_prompt, cb_w, cb_h, cb_custom, voice_radio, voice_role, voice_speed, voice_pit, voice_vol, audition,
            btn_txt_to_voice, output_type, voice_emotion, voice_emotion_intensity, lora_name, batch_images, custom_preset_title, custom_preset,
            add_preset):
        p.do_not_save_grid = True
        # here the logic for saving images in the original sd is disabled
        p.do_not_save_samples = True

        p.batch_size = 1
        # p.n_iter = 1
        processed = process(p, prompt_txt, prompts_folder, int(max_frames), custom_font, text_font_path, text_watermark, text_watermark_color,
                            text_watermark_content, text_watermark_font, text_watermark_pos, text_watermark_size, text_watermark_target, save_or,
                            default_prompt_type, need_default_prompt, need_negative_prompt, need_combine_prompt, combine_prompt_type,
                            cb_h, cb_w, lora_name, int(batch_images))

        return processed
