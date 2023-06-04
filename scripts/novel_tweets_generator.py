import copy
import hashlib
import importlib.util
import json
import os
import random
import re
import shlex
import subprocess
import sys
import traceback
from datetime import datetime

import gradio as gr
import requests
from PIL import Image, ImageDraw, ImageFont
from natsort import natsorted
import modules.scripts as scripts
from modules import images
from modules.processing import process_images
from modules.shared import state
from scripts import prompts_styles as ps
from scripts import voice_params as vop
from revChatGPT.V1 import Chatbot as ChatbotV1
from revChatGPT.V3 import Chatbot as ChatbotV3
from dotenv import load_dotenv
import uuid
from urllib.parse import quote_plus
from urllib.parse import urlencode
import time
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
import base64
import wave


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag == "true") else False


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
os.makedirs(novel_tweets_generator_images_folder, exist_ok=True)
novel_tweets_generator_prompts_folder = "outputs/novel_tweets_generator/prompts"
os.makedirs(novel_tweets_generator_prompts_folder, exist_ok=True)
novel_tweets_generator_audio_folder = "outputs/novel_tweets_generator/audio"
os.makedirs(novel_tweets_generator_audio_folder, exist_ok=True)
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
if sys.platform == 'win32':
    print('当前系统是Windows')
    load_dotenv(dotenv_path=f"{parent_dir}\\.env", override=True, verbose=True)
elif sys.platform == 'linux':
    print('当前系统是Linux')
    load_dotenv(dotenv_path=f"{parent_dir}/.env", override=True, verbose=True)
mac = uuid.getnode()
mac_address = ':'.join(("%012X" % mac)[i:i + 2] for i in range(0, 12, 2))
client = AcsClient(
    os.environ.get('ALIYUN_ACCESSKEY_ID'),
    os.environ.get('ALIYUN_ACCESSKEY_SECRET'),
    "cn-shanghai"
)
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
    print("AI推文是：", f"\n{translated_text}")
    return translated_text


# All the image processing is done in this method
def process(p, prompt_txt, prompts_folder, max_frames, custom_font, text_font_path, text_watermark, text_watermark_color,
            text_watermark_content, text_watermark_font, text_watermark_pos, text_watermark_size, text_watermark_target, save_or,
            default_prompt_type, need_default_prompt, need_negative_prompt, need_combine_prompt, combine_prompt_type, cb_h):
    if prompts_folder == "":
        prompts_folder = os.getcwd() + "/" + novel_tweets_generator_prompts_folder
        prompts_folder = prompts_folder.replace("\\", "/")
        prompts_folder = prompts_folder.replace('"', '')
    prompts_folders = count_subdirectories(prompts_folder)
    count = 0
    for root, dirs, files in os.walk(prompts_folder):
        for file in files:
            count += 1

    frames = []
    results = []
    processed_list = []
    if prompts_folders == 0:
        file_count = len(os.listdir(prompts_folder))
        state.job_count = min(int(file_count * p.n_iter), max_frames * 1)
        filenames = []
        result = dura, first_processed, original_images, processed_images, \
            processed_images2, frames_num, cp, cps = deal_with_single_image(max_frames, p, prompt_txt, prompts_folder,
                                                                            default_prompt_type, need_default_prompt,
                                                                            need_negative_prompt, need_combine_prompt,
                                                                            combine_prompt_type, cb_h)
        frames.append(frames_num)
        filenames.append(os.path.basename(prompts_folder))
        images_post_processing(custom_font, filenames, frames, original_images, cp,
                               first_processed, processed_images,
                               processed_images2, save_or, text_font_path, text_watermark,
                               text_watermark_color,
                               text_watermark_content, text_watermark_font, text_watermark_pos,
                               text_watermark_size,
                               text_watermark_target, cps)
        results.append(result)
    else:
        state.job_count = min(int(count * p.n_iter), max_frames * prompts_folders)
        for file_name in os.listdir(prompts_folder):
            folder_path = os.path.join(prompts_folder, file_name)
            if os.path.isdir(folder_path):
                filenames = []
                result = dura, first_processed, original_images, processed_images, \
                    processed_images2, frames_num, cp, cps = deal_with_single_image(max_frames, p, prompt_txt, folder_path,
                                                                                    default_prompt_type, need_default_prompt,
                                                                                    need_negative_prompt, need_combine_prompt,
                                                                                    combine_prompt_type, cb_h)
                frames.append(frames_num)
                filenames.append(os.path.basename(folder_path))
                images_post_processing(custom_font, filenames, frames, original_images, cp,
                                       first_processed, processed_images,
                                       processed_images2, save_or, text_font_path, text_watermark,
                                       text_watermark_color,
                                       text_watermark_content, text_watermark_font, text_watermark_pos,
                                       text_watermark_size,
                                       text_watermark_target, cps)
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
                           need_combine_prompt, combine_prompt_type, cb_h):
    cps = []
    assert os.path.isdir(prompts_folder), f"关键词文件夹-> '{prompts_folder}' 不存在或不是文件夹."
    prompt_files = natsorted(
        [f for f in os.listdir(prompts_folder) if os.path.isfile(os.path.join(prompts_folder, f))])

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
        "3.美女专属(真人-女)": ps.default_prompts_for_girl,
        "4.五光十色(通用)": ps.default_prompts_colorful,
        "5.高达机甲(通用)": ps.default_prompts_gundam,
        "6.高达衣服(通用)": ps.default_prompts_gundam_clothes,
        "7.软糯糖果(二次元-女)": ps.default_prompts_candy,
        "8.盲盒风格(二次元)": ps.default_prompts_blind_box,
        "9.汉服-唐(真人-女)": ps.default_prompts_hanfu_tang,
        "10.汉服-宋(真人-女)": ps.default_prompts_hanfu_song,
        "11.汉服-明(真人-女)": ps.default_prompts_hanfu_ming,
        "12.汉服-晋(真人-女)": ps.default_prompts_hanfu_jin,
        "13.汉服-汉(真人-女)": ps.default_prompts_hanfu_han,
        "14.时尚女孩(通用偏二)": ps.default_prompts_fashion_girl,
        "15.胶片风格(真人-女)": ps.default_prompts_film_girl,
        "16.胶片风格(真人-女)": ps.default_prompts_pixel,
        "17.敦煌风格(真人-女)": ps.default_prompts_dunhuang,
        "18.A素体机娘(通用偏二)": ps.default_prompts_A_Mecha_REN,
        "19.露西(赛博朋克)": ps.default_prompts_Lucy_Cyberpunk,
        "20.jk制服(真人-女)": ps.default_prompts_jk
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
            copy_p.prompt = f"({individual_prompt.replace('.', ' ')}:1.5), {copy_p.prompt}"
            file_idx += 1
        copy_p.seed = int(random.randrange(4294967294))
        if cb_h:
            copy_p.width = 576
            copy_p.height = 1024
        else:
            copy_p.width = 1024
            copy_p.height = 576
        copy_p.sampler_name = 'DPM++ SDE Karras'
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
    return 0, final_processed, original_images, processed_images, processed_images2, frame_count, copy_p, cps


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
                  frames, cps):
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
            watermarked_path = os.path.join(original_dir, f"{original_filename}")
            watermarked_image.save(watermarked_path)
            img1 = Image.open(watermarked_path)
            watered_images.append(img1)
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
                watermarked_path = os.path.join(original_dir, f"{original_filename}")
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
                           text_watermark_target, cps):
    processed_images_flattened = []
    # here starts the custom image saving logic
    if save_or:
        for i, filename in enumerate(filenames):
            for j, img in enumerate(original_images[i]):
                images_dir = f"{novel_tweets_generator_images_folder}/{formatted_date}/{filename}/original_images"
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)
                images.save_image(img, images_dir, "",
                                  prompt=cps[j].prompt, seed=cps[j].seed, grid=False, p=p,
                                  save_to_dirs=False, info=cps[j].info)
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
                                           text_font_path, p, processed, filenames, frames, cps)
        # After adding the watermark, only the final image will be displayed
        processed_images_flattened = []
        for row in watermarked_images:
            processed_images_flattened += row
        if len(processed_images_flattened) == 1:
            processed.images = processed_images_flattened
        else:
            processed.images = [images.image_grid(processed_images_flattened,
                                                  rows=p.batch_size * p.n_iter)] + processed_images_flattened


def ai_process_article(original_article, scene_number, api_cb, use_proxy):
    proxy = None
    default_pre_prompt = """你是专业的场景转换描述专家，我给你一段文字，并指定你需要转换的场景个数，你需要把他分为不同的场景。每个场景必须要细化，必须要细化环境描写（天气，周围有些什么等等内容），必须要细化人物描写（人物衣服，衣服样式，衣服颜色，表情，动作，头发，发色等等），如果多个场景中出现的人物是同一个，请统一这个任务的衣服，发色等细节。如果场景中出现多个人物，还必须要细化每个人物的细节。
    你回答的场景要加入自己的一些想象，但不能脱离原文太远。你的回答请务必将每个场景的描述转换为单词，并使用多个单词描述场景，每个场景至少6个单词，如果场景中出现了人物,请给我添加人物数量的描述，例如 一个女孩，一个男孩，5个女孩等等。不要用一段话给我回复。请你将我给你的文字转换场景，并且按照这个格式给我：
    1.场景单词1, 场景单词2, 场景单词3, 场景单词4, 场景单词5, 场景单词6, ...
    2.场景单词1, 场景单词2, 场景单词3, 场景单词4, 场景单词5, 场景单词6, ...
    3.场景单词1, 场景单词2, 场景单词3, 场景单词4, 场景单词5, 场景单词6, ...
    ...
    等等
    你只用回复场景内容，其他的不要回复。
    例如这一段话：在未来的世界中，地球上的资源已经枯竭，人类只能依靠太空探索来维持生存。在这个时代，有一位年轻的女子名叫艾米丽，她是一名出色的宇航员，拥有超凡的技能和无畏的勇气。她的目标是在银河系中寻找新的星球，为人类开辟新的家园。
    将它分为三个场景，你需要这样回答我：
    1.未来，世界末日，沙漠，无人，灰色的天空，风
    2.星际飞船，驾驶舱，一个女孩，穿着太空服，坐着，表情平静，美丽，看着操作屏幕
    3.太空，天空，星舰，恒星，行星，银河系
    请你牢记这些规则，任何时候都不要忘记。
    """
    prompt = default_pre_prompt + "\n" + f"内容是：{original_article}\n必须将其转换为{int(scene_number)}个场景"
    response = ""
    if use_proxy:
        proxy = os.environ.get('PROXY')
    if api_cb:
        try:
            openai_key = os.environ.get('KEY')
            chatbot = ChatbotV3(api_key=openai_key, proxy=proxy if (proxy != "" or proxy is not None) else None)
            response = chatbot.ask(prompt=prompt)
        except Exception as e:
            print(f"Error: {e}")
            response = "抱歉，发生了一些意外，请重试。"
    else:
        configs = {
            "access_token": f"{os.environ.get('ACCESS_TOKEN')}",
            "base_url": os.environ.get('CHATGPT_BASE_URL')
        }
        if proxy is not None and proxy != "":
            configs['proxy'] = proxy.replace('http://', '')
        try:
            chatbot = ChatbotV1(config=configs)
            for data in chatbot.ask(prompt):
                response = data["message"]
        except Exception as e:
            print(f"Error: {e}")
            response = "抱歉，发生了一些意外，请重试。"
    return gr.update(value=response), gr.update(interactive=True)


def save_prompts(prompts, is_translate=False):
    if prompts != "":
        print("开始处理并保存AI推文")
        appid = os.environ.get('BAIDU_TRANSLATE_APPID')
        key = os.environ.get('BAIDU_TRANSLATE_KEY')
        if is_translate:
            prompts = baidu_translate(prompts, 'zh', 'en', appid, key)
        current_folders = count_subdirectories(novel_tweets_generator_prompts_folder)
        novel_tweets_generator_prompts_sub_folder = 'outputs/novel_tweets_generator/prompts/' + f'{current_folders + 1}'
        os.makedirs(novel_tweets_generator_prompts_sub_folder, exist_ok=True)
        lines = prompts.splitlines()
        for line in lines:
            parts = line.split()
            content = ' '.join(parts[1:])
            filename = novel_tweets_generator_prompts_sub_folder + "/scene" + parts[0][:-1] + '.txt'
            with open(filename, 'w') as f:
                f.write(content)
        print("AI推文保存完成")
        return gr.update(interactive=True)
    else:
        print("AI处理的推文为空，不做处理")
        return gr.update(interactive=True)


def change_state(is_checked):
    if is_checked:
        return gr.update(value=False)
    else:
        return gr.update(value=True)


def set_un_clickable():
    return gr.update(interactive=False)


def tts_fun(text, spd, pit, vol, per, aue, tts_type, voice_emotion, voice_emotion_intensity):
    print("语音引擎类型是:", tts_type)
    if tts_type == "百度":
        tts_baidu(aue, per, pit, spd, text, vol)
    elif tts_type == "阿里":
        tts_ali(text, spd, pit, vol, per, aue, voice_emotion, voice_emotion_intensity)
    elif tts_type == "华为":
        tts_huawei(aue, per, pit, spd, text, vol)
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
        'client_id': os.environ.get('BAIDU_VOICE_API_KEY'),
        'client_secret': os.environ.get('BAIDU_VOICE_SECRET_KEY'),
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
    app_key = os.environ.get('ALIYUN_APPKEY')
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
            "device_id": mac_address
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
                            "key": os.environ.get('HUAWEI_AK')
                        },
                        "secret": {
                            "key": os.environ.get('HUAWEI_SK')
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
        return gr.update(choices=vop.huawei['voice_role'], value=vop.huawei['voice_role'][0]), gr.update(visible=True), \
            gr.update(minimum=-500, maximum=500, value=0, step=100), gr.update(minimum=-500, maximum=500, value=0, step=100), \
            gr.update(minimum=0, maximum=100, value=50, step=10), gr.update(choices=vop.huawei['aue'], value=vop.huawei['aue'][0]), \
            gr.update(visible=False), gr.update(visible=False)


def change_voice_role(tts_type, role):
    if tts_type == '阿里':
        emo_zh = []
        if '多情感' in role:
            for key in role_dict:
                if key in role:
                    emo_zh, emo_en = role_dict[key]
            return gr.update(choices=emo_zh, value=emo_zh[0], visible=True), gr.update(minimum=0.01, maximum=2.0, value=1, visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)
    elif tts_type == '百度':
        return gr.update(visible=False), gr.update(visible=False)
    elif tts_type == '华为':
        return gr.update(visible=False), gr.update(visible=False)


class Script(scripts.Script):

    def title(self):
        return "小说推文图片生成器"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        gr.HTML("此脚本可以与controlnet一起使用，若一起使用请把controlnet的参考图留空。")
        with gr.Accordion(label="基础属性，必填项，每一项都不能为空", open=True):
            with gr.Column(variant='panel'):
                with gr.Accordion(label="1. 默认提示词相关", open=True):
                    default_prompt_type = gr.Dropdown(
                        [
                            "1.基本提示(通用)", "2.基本提示(通用修手)", "3.美女专属(真人-女)", "4.五光十色(通用)",
                            "5.高达机甲(通用)", "6.高达衣服(通用)",
                            "7.软糯糖果(二次元-女)", "8.盲盒风格(二次元)", "9.汉服-唐(真人-女)", "10.汉服-宋(真人-女)",
                            "11.汉服-明(真人-女)", "12.汉服-晋(真人-女)", "13.汉服-汉(真人-女)",
                            "14.时尚女孩(通用偏二)",
                            "15.胶片风格(真人-女)", "16.像素风格(二次元)", "17.敦煌风格(真人-女)",
                            "18.A素体机娘(通用偏二)",
                            "19.露西(赛博朋克)", "20.jk制服(真人-女)"
                        ],
                        label="默认正面提示词类别",
                        value="1.基本提示(通用)")
                    need_combine_prompt = gr.Checkbox(label="需要组合技(组合上方类别)？", value=False)
                    combine_prompt_type = gr.Textbox(
                        label="请输入你需要组合的类别组合，例如2+3+4，不要组合过多种类",
                        visible=False)

                    def is_show_combine(is_show):
                        return gr.update(visible=is_show)

                    need_combine_prompt.change(is_show_combine, inputs=[need_combine_prompt],
                                               outputs=[combine_prompt_type])
                    with gr.Row():
                        need_default_prompt = gr.Checkbox(label="自行输入默认正面提示词(勾选后上面选择将失效)",
                                                          value=False)
                        need_negative_prompt = gr.Checkbox(label="自行输入默认负面提示词(勾选后需要自行输入负面提示词)",
                                                           value=False)

                    prompt_txt = gr.Textbox(label="默认提示词，将影响各帧", lines=3, max_lines=5, value="",
                                            visible=False)

                    def is_need_default_prompt(is_need):
                        return gr.update(visible=is_need)

                    need_default_prompt.change(is_need_default_prompt, inputs=[need_default_prompt],
                                               outputs=[prompt_txt])

                    need_mix_models = gr.Checkbox(label="分片使用模型(勾选后请在下方输入模型全名包括后缀,使用 ',' 分隔)",
                                                  value=False, visible=False)
                    models_txt = gr.Textbox(label="模型集合", lines=3, max_lines=5, value="",
                                            visible=False)

                    def is_need_mix_models(is_need):
                        return gr.update(visible=is_need)

                    need_mix_models.change(is_need_mix_models, inputs=[need_mix_models],
                                           outputs=[models_txt])
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
                    scene_number = gr.Number(
                        label="输入要生成的场景数量",
                        value=10,
                        min=1
                    )
                    with gr.Row():
                        api_cb = gr.Checkbox(
                            label="使用api方式处理",
                            info="需要填写KEY",
                            value=True
                        )
                        web_cb = gr.Checkbox(
                            label="使用web方式处理",
                            info="需要填写ACCESS_TOKEN"
                        )
                        cb_use_proxy = gr.Checkbox(
                            label="使用代理",
                            info="需要填写PROXY"
                        )
                        cb_trans_prompt = gr.Checkbox(
                            label="翻译AI推文,保存推文时生效",
                            info="需要填写百度翻译的appid和key"
                        )
                        api_cb.change(change_state, inputs=[api_cb], outputs=[web_cb])
                        web_cb.change(change_state, inputs=[web_cb], outputs=[api_cb])
                    ai_article = gr.Textbox(
                        label="AI处理的推文将显示在这里",
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
                    deal_with_ai.click(ai_process_article, inputs=[original_article, scene_number, api_cb, cb_use_proxy],
                                       outputs=[ai_article, deal_with_ai])
                    btn_save_ai_prompts.click(set_un_clickable, outputs=[btn_save_ai_prompts])
                    btn_save_ai_prompts.click(save_prompts, inputs=[ai_article, cb_trans_prompt], outputs=[btn_save_ai_prompts])
                with gr.Accordion(label="3.2 原文转语音"):
                    # 其他语音合成引擎 '华为', '微软', '谷歌' 待开发
                    voice_radio = gr.Radio(['百度', '阿里', '华为'], label="3.2.1 语音引擎", info='请选择一个语音合成引擎，默认为百度', value='百度')
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
                    voice_role.change(change_voice_role, inputs=[voice_radio, voice_role], outputs=[voice_emotion, voice_emotion_intensity])
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
                                                   voice_emotion, voice_emotion_intensity],
                                           outputs=[btn_txt_to_voice])
                prompts_folder = gr.Textbox(
                    label="4. 输入包含提示词文本文件的文件夹路径",
                    info="默认值为空时处理outputs/novel_tweets_generator/prompts文件夹下的所有文件夹",
                    lines=1,
                    max_lines=2,
                    value=""
                )
                gr.HTML("5. 生成图片类型")
                with gr.Row():
                    cb_h = gr.Checkbox(label='竖图', value=True, info="尺寸为576*1024，即9:16的比例")
                    cb_w = gr.Checkbox(label='横图', info="尺寸为1024*576，即16:9的比例")
                    cb_h.change(change_state, inputs=[cb_h], outputs=[cb_w])
                    cb_w.change(change_state, inputs=[cb_w], outputs=[cb_h])
                gr.HTML("")

            with gr.Accordion(label="去除背景和保留原图(至少选择一项否则文件夹中没有保留生成的图片)", open=True, visible=False):
                with gr.Column(variant='panel', visible=False):
                    with gr.Row():
                        save_or = gr.Checkbox(label="8. 是否保留原图",
                                              info="为了不影响查看原图，默认选中会保存未删除背景的图片", value=True)

            with gr.Accordion(label="更多操作(打开看看说不定有你想要的功能)", open=False):
                with gr.Column(variant='panel'):
                    with gr.Column():
                        text_watermark = gr.Checkbox(label="5. 添加文字水印", info="自定义文字水印")
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

        return [prompt_txt, max_frames, prompts_folder, save_or, text_watermark, text_watermark_font, text_watermark_target,
                text_watermark_pos, text_watermark_color, text_watermark_size, text_watermark_content, custom_font, text_font_path,
                default_prompt_type, need_default_prompt, need_negative_prompt, need_combine_prompt, combine_prompt_type, original_article,
                scene_number, deal_with_ai, ai_article, api_cb, web_cb, btn_save_ai_prompts,
                tb_save_ai_prompts_folder_path, cb_use_proxy, cb_trans_prompt, cb_w, cb_h, voice_radio, voice_role, voice_speed, voice_pit, voice_vol,
                audition, btn_txt_to_voice, output_type, voice_emotion, voice_emotion_intensity]

    def run(self, p, prompt_txt, max_frames, prompts_folder, save_or, text_watermark, text_watermark_font, text_watermark_target,
            text_watermark_pos, text_watermark_color, text_watermark_size, text_watermark_content, custom_font, text_font_path, default_prompt_type,
            need_default_prompt, need_negative_prompt, need_combine_prompt, combine_prompt_type,
            original_article, scene_number, deal_with_ai, ai_article, api_cb, web_cb, btn_save_ai_prompts, tb_save_ai_prompts_folder_path,
            cb_use_proxy, cb_trans_prompt, cb_w, cb_h, voice_radio, voice_role, voice_speed, voice_pit, voice_vol, audition, btn_txt_to_voice,
            output_type, voice_emotion, voice_emotion_intensity):
        p.do_not_save_grid = True
        # here the logic for saving images in the original sd is disabled
        p.do_not_save_samples = True

        p.batch_size = 1
        p.n_iter = 1
        processed = process(p, prompt_txt, prompts_folder, int(max_frames), custom_font, text_font_path, text_watermark, text_watermark_color,
                            text_watermark_content, text_watermark_font, text_watermark_pos, text_watermark_size, text_watermark_target, save_or,
                            default_prompt_type, need_default_prompt, need_negative_prompt, need_combine_prompt, combine_prompt_type,
                            cb_h)

        return processed
