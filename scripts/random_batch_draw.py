import modules.scripts as scripts
import gradio as gr
from PIL import Image, ImageSequence, ImageDraw, ImageFont
from modules import images
from modules.processing import process_images
from modules.shared import state
from threading import Thread
import traceback
import subprocess
import sys
import importlib.util
import shlex
import copy
import hashlib
import random
import requests
import json
import chardet
import re
import argparse
import os
import tempfile
import urllib
from scripts import prompts as pt
import threading


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


def run_pip(args, desc=None):
    index_url = os.environ.get('INDEX_URL', "")
    python = sys.executable
    index_url_line = f' --index-url {index_url}' if index_url != '' else ''
    return run(f'"{python}" -m pip {args} -i https://pypi.douban.com/simple', desc=f"Installing {desc}",
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


def random_weights(num_weights, min_weight=0.1, max_sum=1.0, special_index=None, special_min=None, special_max=None):
    while True:
        weights = [round(random.uniform(min_weight, max_sum - min_weight * (num_weights - 1)), 1)
                   for _ in range(num_weights - 1)]
        if special_index is not None:
            special_weight = round(random.uniform(special_min, special_max), 1)
            weights.insert(special_index, special_weight)
        weights_sum = sum(weights)
        last_weight = round(max_sum - weights_sum, 1)
        if min_weight <= last_weight <= max_sum:
            weights.append(last_weight)
            return weights


def random_prompt_selection(prompt_lists):
    selected_prompts = []

    # 随机选择 action_prompts 或 actions_prompts 中的一个元素
    action_and_actions = prompt_lists[-4:-2]
    selected_action_list = random.choice(action_and_actions)
    selected_action = random.choice(selected_action_list) if isinstance(selected_action_list[0],
                                                                        str) else random.choice(
        random.choice(selected_action_list))

    # 随机选择 clothes_prompts 或 clothes_prompts2 中的一个元素
    clothes_and_clothes2 = prompt_lists[-2:]
    selected_clothes_list = random.choice(clothes_and_clothes2)
    selected_clothes = random.choice(selected_clothes_list) if isinstance(selected_clothes_list[0],
                                                                          str) else random.choice(
        random.choice(selected_clothes_list))

    # 其他的prompt列表
    other_prompts = prompt_lists[:-4]

    for prompt_list in other_prompts:
        if isinstance(prompt_list[0], list):
            combined_prompts = ", ".join([random.choice(sub_list) for sub_list in prompt_list])
            selected_prompts.append(combined_prompts)
        else:
            selected_prompts.append(random.choice(prompt_list))

    # 将随机选择的 action 和 clothes 添加到结果中
    selected_prompts.append(selected_action)
    selected_prompts.append(selected_clothes)

    return ", ".join(selected_prompts)


# analyzing parameters by imitating mj
def parse_args(args_str, default=None):
    parser = argparse.ArgumentParser()
    arg_dict = {}
    arg_name = None
    arg_values = []
    if args_str != "":
        for arg in args_str.split():
            if arg.startswith("--"):
                if arg_name is not None:
                    if len(arg_values) == 1:
                        arg_dict[arg_name] = arg_values[0]
                    elif len(arg_values) > 1:
                        arg_dict[arg_name] = arg_values[0:]
                    else:
                        arg_dict[arg_name] = default
                arg_name = arg.lstrip("-")
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


def assign_scene(images_num, scene_num, current_num, scenes):
    if images_num > 1:
        images_per_scene = images_num // scene_num
        scene_num = int(scene_num)
        for i in range(scene_num):
            if current_num < images_per_scene * (i + 1):
                return scenes[i]
    return scenes[-1] if images_num > 1 else scenes[0]


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


def mcprocess(p, images_num, scene1, scene2, scene3, scene4, scene5, scene6, scene7, scene8, scene9, scene10,
              content_num, is_img2img):
    first_processed = None
    original_images = []
    cps = []
    parsed_args = {}
    scenes = [scene1, scene2, scene3, scene4, scene5, scene6, scene7, scene8, scene9, scene10]
    for i in range(p.batch_size * p.n_iter):
        original_images.append([])

    p.do_not_save_grid = True

    state.job_count = int(images_num * p.n_iter)
    frame_count = 0

    copy_p = copy.copy(p)
    prompts_lists = [pt.camera_perspective_prompts, pt.person_prompts, pt.career_prompts,
                     pt.facial_features_prompts, pt.light_prompts,
                     pt.expression_prompts, pt.hair_prompts, pt.decoration_prompts, pt.hat_prompts,
                     pt.shoes_prompts, pt.socks_prompts, pt.gesture_prompt, pt.sight_prompts, pt.environment_prompts,
                     pt.style_prompts, pt.action_prompts, pt.actions_prompts, pt.clothes_prompts, pt.clothes_prompts2
                     ]
    anime_prompts_lists = [pt.anime_characters_prompts, pt.camera_perspective_prompts, pt.person_prompts,
                           pt.career_prompts, pt.facial_features_prompts, pt.expression_prompts, pt.hair_prompts,
                           pt.decoration_prompts, pt.hat_prompts, pt.shoes_prompts, pt.socks_prompts, pt.gesture_prompt,
                           pt.sight_prompts, pt.environment_prompts, pt.light_prompts, pt.style_prompts, pt.action_prompts,
                           pt.actions_prompts, pt.clothes_prompts, pt.clothes_prompts2
                           ]

    lora_prompts = ['lora:cuteGirlMix4_v10', 'lora:koreandolllikenessV20_v20', 'lora:taiwanDollLikeness_v20',
                    'lora:japanesedolllikenessV1_v15']
    special_index = lora_prompts.index('lora:cuteGirlMix4_v10')
    lora_weights = random_weights(len(lora_prompts), special_index=special_index, special_min=0.4,
                                  special_max=0.6)
    combined_lora_prompts_string = ", ".join([f"<{prompt}:{weight}>" for prompt, weight in zip(lora_prompts,
                                                                                               lora_weights)])
    for num in range(images_num):
        is_real = False
        add_random_prompts = False
        add_styles = False
        scene = assign_scene(images_num, content_num, num, scenes)
        if scene != "":
            parsed_args = parse_args(scene)
            index = scene.find("--")
            if index != -1:
                scene = scene[:index]

        if state.interrupted:
            state.nextjob()
            break
        if state.skipped:
            state.skipped = False
        state.job = f"{state.job_no + 1} out of {state.job_count}"

        other_prompts = ''

        # deal with args
        if bool(parsed_args):
            if 'real' in parsed_args:
                is_real = True
            if 'arp' in parsed_args:
                add_random_prompts = True
            if 'ar' in parsed_args:
                ar_value = parsed_args.get('ar')
                if ar_value == '1:1':
                    copy_p.width = 512
                    copy_p.height = 512
                if ar_value == '3:4':
                    copy_p.width = 600
                    copy_p.height = 800
                if ar_value == '4:3':
                    copy_p.width = 800
                    copy_p.height = 600
                if ar_value == '9:16':
                    copy_p.width = 540
                    copy_p.height = 960
                if ar_value == '16:9':
                    copy_p.width = 960
                    copy_p.height = 540
            if 'rf' in parsed_args:
                copy_p.restore_faces = True
            if 'sn' in parsed_args:
                sn_value = parsed_args.get('sn')
                if isinstance(sn_value, list):
                    combined_sn = ' '.join(sn_value)
                    copy_p.sampler_name = combined_sn
                elif isinstance(sn_value, str):
                    copy_p.sampler_name = sn_value
            if 'cs' in parsed_args:
                cs_value = float(parsed_args.get('cs'))
                copy_p.cfg_scale = cs_value
            if 'steps' in parsed_args:
                steps_value = int(parsed_args.get('steps'))
                copy_p.steps = steps_value
            if 'seed' in parsed_args:
                seed_value = parsed_args.get('seed')
                copy_p.seed = seed_value
            if 'tl' in parsed_args:
                copy_p.tiling = True
            if 'as' in parsed_args:
                add_styles = True
            if is_img2img:
                if 'ds' in parsed_args:
                    ds_value = float(parsed_args.get('ds'))
                    copy_p.denoising_strength = ds_value
            if 'img' in parsed_args:
                img_value = parsed_args.get('img')
                if img_value.startswith('http') or img_value.startswith('https'):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        filepath, headers = urllib.request.urlretrieve(img_value,
                                                                       os.path.join(temp_dir, 'tmp_image.jpg'))
                        tmp_image = Image.open(filepath)
                else:
                    img_value = img_value.replace("\\", "/")
                    img_value = img_value.replace('"', '')
                    tmp_image = Image.open(img_value)
                copy_p.init_images = [tmp_image]
            # hr is only in t2i
            if not is_img2img:
                if 'hr' in parsed_args:
                    hr_value = parsed_args.get('hr')
                    copy_p.enable_hr = True
                    if isinstance(hr_value, list):
                        if len(hr_value) == 4:
                            copy_p.hr_upscaler = hr_value[0]
                        elif len(hr_value) == 5:
                            copy_p.hr_upscaler = ' '.join(hr_value[:2])
                        elif len(hr_value) == 6:
                            copy_p.hr_upscaler = ' '.join(hr_value[:3])
                        copy_p.hr_second_pass_steps = int(hr_value[-3])
                        copy_p.denoising_strength = float(hr_value[-2])
                        copy_p.hr_scale = float(hr_value[-1])

        if is_real:
            other_prompts = random_prompt_selection(prompts_lists)
            copy_p.negative_prompt = pt.real_person_negative_prompt
        else:
            other_prompts = random_prompt_selection(anime_prompts_lists)
            copy_p.negative_prompt = pt.anime_negative_prompt

        if scene != "":
            if add_random_prompts:
                if is_real:
                    copy_p.prompt = f"{scene}, {pt.default_prompt}, mix4, {combined_lora_prompts_string}, " \
                                    f"{other_prompts}"
                else:
                    copy_p.prompt = f"{scene}, {pt.default_prompt}, {other_prompts}"
            else:
                if is_real:
                    copy_p.prompt = f"{scene}, {pt.default_prompt}, mix4, {combined_lora_prompts_string}"
                else:
                    copy_p.prompt = f"{scene}, {pt.default_prompt}"
        else:
            if is_real:
                copy_p.prompt = f"{pt.default_prompt}, mix4, {combined_lora_prompts_string}, " \
                                f"{other_prompts}"
            else:
                copy_p.prompt = f"{pt.default_prompt}, {other_prompts}"
        copy_p.seed = int(random.randrange(4294967294))
        processed = process_images(copy_p)

        cps.append(processed)
        for i, img1 in enumerate(processed.images):
            if i > 0:
                break
            original_images[i].append(img1)
        frame_count += 1
    # 这里仅仅是为了处理显示出来的提示词和图片不一致的问题
    copy_cp = copy.deepcopy(cps)
    final_processed = merge_processed_objects(cps)
    if len(cps) > 1:  # 只有一张图片的时候不做插入数据的操作
        copy_cp.insert(0, process_images(p))  # 插入一个空白数据为了解决网页显示的第一个图片是宫格图的时候造成后面的图片信息异常的问题
        final_processed = merge_processed_objects(copy_cp)
    return original_images, final_processed


class Script(scripts.Script):

    def title(self):
        return "随机图片生成器"

    def ui(self, is_img2img):
        with gr.Accordion(label="随机做点图看看吧", open=True):
            with gr.Column():
                images_num = gr.Number(label="请输入要作图的数量", value=1, min=1)
                scene1 = gr.Textbox(label="请输入你想要的内容，当然你喜欢抽盲盒的话可以什么也不填哦", value='', lines=1,
                                    max_lines=2)
                scene2 = gr.Textbox(label="请输入你想要的内容2，当然你喜欢抽盲盒的话可以什么也不填哦", value='', lines=1,
                                    max_lines=2, visible=False)
                scene3 = gr.Textbox(label="请输入你想要的内容3，当然你喜欢抽盲盒的话可以什么也不填哦", value='', lines=1,
                                    max_lines=2, visible=False)
                scene4 = gr.Textbox(label="请输入你想要的内容4，当然你喜欢抽盲盒的话可以什么也不填哦", value='', lines=1,
                                    max_lines=2, visible=False)
                scene5 = gr.Textbox(label="请输入你想要的内容5，当然你喜欢抽盲盒的话可以什么也不填哦", value='', lines=1,
                                    max_lines=2, visible=False)
                scene6 = gr.Textbox(label="请输入你想要的内容6，当然你喜欢抽盲盒的话可以什么也不填哦", value='', lines=1,
                                    max_lines=2, visible=False)
                scene7 = gr.Textbox(label="请输入你想要的内容7，当然你喜欢抽盲盒的话可以什么也不填哦", value='', lines=1,
                                    max_lines=2, visible=False)
                scene8 = gr.Textbox(label="请输入你想要的内容8，当然你喜欢抽盲盒的话可以什么也不填哦", value='', lines=1,
                                    max_lines=2, visible=False)
                scene9 = gr.Textbox(label="请输入你想要的内容9，当然你喜欢抽盲盒的话可以什么也不填哦", value='', lines=1,
                                    max_lines=2, visible=False)
                scene10 = gr.Textbox(label="请输入你想要的内容10，当然你喜欢抽盲盒的话可以什么也不填哦", value='', lines=1,
                                     max_lines=2, visible=False)

                scene_num = gr.Number(label="想要的内容数量", value=1, min=1, visible=False, interactive=False)
                with gr.Row():
                    btn_add = gr.Button(value="添加内容输入框")
                    btn_rem = gr.Button(value="移除内容输入框", visible=False)

                    def update_content_visible(num, operation):
                        num = int(num)

                        if operation == "add":
                            num += 1
                        elif operation == "subtract":
                            num -= 1

                        btn_add_visible = num < 10
                        btn_rem_visible = num > 1

                        visible_states = [False] * 10
                        for i in range(num):
                            visible_states[i] = True

                        return num, gr.update(visible=btn_add_visible), gr.update(visible=btn_rem_visible), *[gr.update(visible=state) for state in
                                                                                                              visible_states]

                    def on_add_click(num):
                        return update_content_visible(num, "add")

                    def on_subtract_click(num):
                        return update_content_visible(num, "subtract")

                    outputs = [scene_num, btn_add, btn_rem] + [scene1, scene2, scene3, scene4, scene5, scene6, scene7, scene8, scene9, scene10]
                    btn_add.click(on_add_click, inputs=[scene_num], outputs=outputs)
                    btn_rem.click(on_subtract_click, inputs=[scene_num], outputs=outputs)

                info = gr.HTML("<br>声明：！！！本脚本只提供批量作图功能，使用者做的图与脚本作者本人无关！！！")
        return [images_num, scene1, scene2, scene3, scene4, scene5, scene6, scene7, scene8, scene9, scene10, scene_num, btn_add, btn_rem, info]

    def run(self, p, images_num, scene1, scene2, scene3, scene4, scene5, scene6, scene7, scene8, scene9, scene10, scene_num, btn_add, btn_rem,
            info):

        p.do_not_save_grid = True

        p.batch_size = 1
        p.n_iter = 1
        original_images, processed = mcprocess(p, int(images_num), scene1, scene2, scene3, scene4, scene5, scene6, scene7, scene8, scene9, scene10,
                                               scene_num, self.is_img2img)

        p.prompt_for_display = processed.prompt
        processed_images_flattened = []

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

        return processed
