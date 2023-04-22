import modules.scripts as scripts
import gradio as gr
import random
import os

from PIL import Image, ImageSequence, ImageDraw, ImageFont
from modules import images
from modules.processing import process_images
from modules.shared import state
import re
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


def mcprocess(p, prompt_txt, file_txt, jump, use_individual_prompts, prompts_folder, max_frames, enable_translate,
              appid, secret_key):
    assert os.path.isdir(prompts_folder), f"关键词文件夹-> '{prompts_folder}' 不存在或不是文件夹."
    prompt_files = sorted(
        [f for f in os.listdir(prompts_folder) if os.path.isfile(os.path.join(prompts_folder, f))])

    first_processed = None
    original_images = []
    processed_images = []
    processed_images2 = []
    for i in range(p.batch_size * p.n_iter):
        original_images.append([])
        processed_images.append([])
        processed_images2.append([])

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

    jumps = int(jump)
    state.job_count = min(int(len(prompt_files) * p.n_iter / jumps), max_frames)

    j = -1
    file_idx = 0
    frame_count = 0

    copy_p = copy.copy(p)
    for prompt_file in prompt_files:
        if state.interrupted:
            state.nextjob()
            break
        if state.skipped:
            state.skipped = False
        state.job = f"{state.job_no + 1} out of {state.job_count}"
        j = j + 1
        if j % jumps != 0:
            continue
        if frame_count >= max_frames:
            break
        for k, v in args.items():
            setattr(copy_p, k, v)

        if file_idx < len(prompt_files):
            prompt_file = os.path.join(prompts_folder, prompt_files[file_idx])
            # 打开文件，获取文件编码
            with open(prompt_file, "rb") as f:
                result = chardet.detect(f.read())
                file_encoding = result['encoding']
            print("当前文件编码格式：", file_encoding)
            with open(prompt_file, "r", encoding=file_encoding) as f:
                individual_prompt = f.read().strip()
            if enable_translate:
                copy_p.prompt = baidu_translate(f"{individual_prompt} {copy_p.prompt}", 'zh', 'en', appid, secret_key)
            else:
                copy_p.prompt = f"{individual_prompt} {copy_p.prompt}"
            file_idx += 1
        else:
            print(f"Warning: 输入的提示词文件数量不足,后续图片生成将只使用默认提示词.")

        processed = process_images(copy_p)
        if first_processed is None:
            first_processed = processed

        for i, img1 in enumerate(processed.images):
            if i > 0:
                break
            original_images[i].append(img1)
        frame_count += 1

    return original_images, first_processed, processed_images, processed_images2, 0


# 添加文字水印
def add_watermark(need_add_watermark_images, need_add_watermark_images1, new_images, or_images,
                  text_watermark_color, text_watermark_content, text_watermark_pos, text_watermark_target,
                  text_watermark_size, text_watermark_font, custom_font, text_font_path, p, processed):
    # 默认字体 微软雅黑
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
    # 将16进制的颜色改为RGBA的颜色
    fill = tuple(int(text_watermark_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
    # 这里存放临时图片
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

    for i, img in enumerate(need_add_watermark_images):

        if int(text_watermark_target) == 0:
            # 这里只是为了拿到宽高，text_overlay_image在int(text_watermark_target) == 0无意义
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
            (fullfn, _) = images.save_image(img, p.outpath_samples, "tmp",
                                            prompt=p.prompt_for_display, seed=processed.seed, grid=False, p=p)
            # 临时图片
            tmp_images.append(fullfn)
            original_dir, original_filename = os.path.split(fullfn)
            original_image = Image.open(fullfn)
            # 将原始图片转换为RGBA格式
            original_image = original_image.convert("RGBA")
            watermarked_image = Image.alpha_composite(original_image, text_overlay_image)
            original_filename = original_filename.replace("tmp-", "")

        else:
            watermarked_image = Image.alpha_composite(bg.convert('RGBA'), text_overlay_image)
            original_dir, original_filename = os.path.split(need_add_watermark_images[i][0])

        watermarked_path = os.path.join(original_dir, f"watermarked_{original_filename}")
        watermarked_image.save(watermarked_path)
        img1 = Image.open(watermarked_path)
        watered_images.append(img1)
    if int(text_watermark_target) == 2:
        for i, img in enumerate(need_add_watermark_images1):
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
            # 这里由于拿到的是内存中的图片没有地址，需要先保存再删除,
            (fullfn, _) = images.save_image(img, p.outpath_samples, "tmp",
                                            prompt=p.prompt_for_display, seed=processed.seed, grid=False, p=p)
            # 临时图片
            tmp_images1.append(fullfn)
            original_dir, original_filename = os.path.split(fullfn)
            # 打开原始图片
            original_image = Image.open(fullfn)
            width, height = original_image.size
            # 将原始图片转换为RGBA格式
            original_image = original_image.convert("RGBA")
            # 创建一个与原始图片相同尺寸的透明图层
            transparent_layer = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(transparent_layer)
            # 在透明图层上绘制水印文字
            draw.text((x, y), text_watermark_content, font=font, fill=fill)
            # 将透明图层叠加在原始图片上
            watermarked_image = Image.alpha_composite(original_image, transparent_layer)
            # 保存添加了水印的图片
            original_filename = original_filename.replace("tmp-", "")
            watermarked_path = os.path.join(original_dir, f"watermarked_{original_filename}")
            watermarked_image.save(watermarked_path)
            img1 = Image.open(watermarked_path)
            watered_images.append(img1)
    # 删除临时图片
    for path in tmp_images:
        os.remove(path)
    for path in tmp_images1:
        os.remove(path)
    for i, img in enumerate(watered_images):
        watermarked_images[i].append(img)
    return watermarked_images


def baidu_translate(query, from_lang, to_lang, appid, secret_key):

    # Step1. 将请求参数中的 appid、翻译 query、随机数 salt、密钥的顺序拼接得到字符串 sign_str
    salt = random.randint(32768, 65536)
    sign_str = appid + query + str(salt) + secret_key
    # Step2. 对 sign_str 进行 MD5 加密，得到 32 位小写的 sign
    sign = hashlib.md5(sign_str.encode('utf-8')).hexdigest()

    # 构造请求url
    url = f'https://fanyi-api.baidu.com/api/trans/vip/translate?q={query}&from={from_lang}&to={to_lang}' \
          f'&appid={appid}&salt={salt}&sign={sign}'

    # 发送请求，获得翻译结果
    resp = requests.get(url)
    result = json.loads(resp.text)
    translated_text = result['trans_result'][0]['dst']
    return translated_text


class Script(scripts.Script):

    def title(self):
        return "批量文生图"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        gr.HTML("此脚本可以与controlnet一起使用")
        gr.HTML("<br>")
        with gr.Accordion(label="基础属性，必填项，每一项都不能为空", open=True):
            with gr.Column(variant='panel'):
                jump = gr.Dropdown(["1", "2", "3", "4", "5"], label="1. 跳帧(1为不跳，2为两帧取一......)", value="1",
                                   visible=False)
                prompt_txt = gr.Textbox(label="1. 默认提示词，将影响每一张图片", lines=3, max_lines=5, value="")
                prompts_folder = gr.Textbox(
                    label="2. 输入每个画面的提示词文本的文件夹路径，每个画面描述的越详细AI越能给你想要的图片",
                    lines=1,
                    max_lines=2,
                    value=""
                )
                file_txt = gr.Textbox(
                    label="3. 输入gif文件路径(如d:\\xxx\\xx.gif,勿出现中文)",
                    lines=1,
                    max_lines=2,
                    visible=False,
                    value="")
                max_frames = gr.Number(
                    label="3. 输入指定的作图数量，用于用户测试生成图片的效果，最小1,测试完成后请输入一个很大的数保证能把所有提示词用完)",
                    value=1000,
                    min=1
                )
        # with gr.Accordion(label="附加选项，根据需要使用", open=False):
        with gr.Column(variant='panel', visible=False):
            use_individual_prompts = gr.Checkbox(
                label="5. 为每一帧选择一个提示词文件（非必选） ",
                value=True
            )

        with gr.Accordion(label="去除背景和保留原图(至少选择一项否则文件夹中没有保留生成的图片)", open=True,
                          visible=False):
            with gr.Column(variant='panel'):
                with gr.Row():
                    rm_bg = gr.Checkbox(label="6. 去除图片的背景仅保留人物?",
                                        info="需要安装rembg，若未安装请点击下方按钮安装rembg")
                    save_or = gr.Checkbox(label="7. 是否保留原图",
                                          info="为了不影响查看原图，默认选中会保存未删除背景的图片", value=True)
                btn_install_rembg = gr.Button(value="安装 rembg").style(full_width=True)

                add_bg = gr.Checkbox(
                    label="为透明图片自定义背景图片",
                    value=False
                )
                bg_path = gr.Textbox(
                    label="输入自定义背景图片路径(勾选上方单选框以及功能6起效)",
                    lines=1, max_lines=2,
                    value=""
                )

        with gr.Accordion(label="附加选项，根据需要使用", open=False):
            with gr.Column(variant='panel', visible=False):
                resize_input = gr.Checkbox(label="8. 调整输入GIF尺寸",
                                           info="注意尺寸修改后根据情况您需要修改图生图的宽高")
                resize_dir = gr.Textbox(
                    label="输入改变尺寸后的gif文件路径(如d:\\xxx\\xx.gif,勿出现中文)，过程有些许耗时,若勾选了上面调整尺寸这个文本为空将会使用输入路径，这样会覆盖原有的GIF图片",
                    lines=1, max_lines=2, value="")
                with gr.Row():
                    width_input = gr.Number(label="调整后的宽度", value=0, min=0)
                    height_input = gr.Number(label="调整后的高度", value=0, min=0)
            # 暂时不开放此功能
            with gr.Row(visible=False):
                resize_output = gr.Checkbox(
                    label="9. 调整输出图片的尺寸(!!!不建议使用!!!不建议使用!!!不建议使用!!!)",
                    info="注意：这里的尺寸修改是直接缩放，可能会影响显示图片效果，无损缩放请使用webui的附加功能进行缩放")

                resize_target = gr.Dropdown(["0", "1", "2", "3"],
                                            label="调整尺寸对象(0:无,1:原始,2:处理过的,3:全部)",
                                            value="0")
            with gr.Row(visible=False):
                width_output = gr.Number(label="调整后的宽度", value=0, min=0)
                height_output = gr.Number(label="调整后的高度", value=0, min=0)
            # 暂时不开放此功能
            with gr.Row(visible=False):
                with gr.Column():
                    make_a_gif = gr.Checkbox(label="10. 合成 GIF 动图(grids文件夹下)", info="可配合帧速率控制选项")
                with gr.Column():
                    frame_rate = gr.Number(
                        label="输出 GIF 的帧速率 (帧/秒),需启用合成 GIF 动图",
                        value=1,
                        min=1,
                        step=1
                    )
                with gr.Column():
                    reverse_gif = gr.Checkbox(label="反转 GIF（倒放 GIF）", info="需启用合成 GIF 动图")
            with gr.Column(variant='panel'):
                with gr.Column():
                    baidu_info = gr.HTML(
                        "<br><a href=https://fanyi-api.baidu.com/doc/11><font "
                        "color=blue>点击这里了解如何获取百度翻译的appid和key</font></a>")
                    enable_translate = gr.Checkbox(label="4. 翻译提示词", info="启用百度翻译,目前仅仅设置了中文翻译为英文")
                    with gr.Row():
                        appid = gr.Textbox(
                            label="百度翻译的APPID",
                            lines=1,
                            max_lines=2,
                            value=""
                        )
                        secret_key = gr.Textbox(
                            label="百度翻译的SECRET_KEY",
                            lines=1,
                            max_lines=2,
                            value=""
                        )
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

        return [jump, prompt_txt, file_txt, max_frames, use_individual_prompts, prompts_folder, rm_bg, save_or,
                btn_install_rembg, resize_input, resize_dir, width_input, height_input, resize_output, resize_target,
                width_output, height_output, make_a_gif, frame_rate, reverse_gif, text_watermark, text_watermark_font,
                text_watermark_target, text_watermark_pos, text_watermark_color, text_watermark_size,
                text_watermark_content, custom_font, text_font_path, add_bg, bg_path, baidu_info, enable_translate,
                appid, secret_key]

    def run(self, p, jump, prompt_txt, file_txt, max_frames, use_individual_prompts, prompts_folder, rm_bg, save_or,
            btn_install_rembg, resize_input, resize_dir, width_input, height_input, resize_output, resize_target,
            width_output, height_output, make_a_gif, frame_rate, reverse_gif, text_watermark, text_watermark_font,
            text_watermark_target, text_watermark_pos, text_watermark_color, text_watermark_size,
            text_watermark_content, custom_font, text_font_path, add_bg, bg_path, baidu_info, enable_translate, appid,
            secret_key):

        if p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        p.do_not_save_grid = True

        p.do_not_save_samples = not save_or

        p.batch_size = 1
        p.n_iter = 1
        original_images, processed, processed_images, processed_images2, dura = mcprocess(
            p, prompt_txt, file_txt, jump, use_individual_prompts, prompts_folder, int(max_frames), enable_translate,
            appid, secret_key)

        p.prompt_for_display = processed.prompt
        processed_images_flattened = []
        if save_or:
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

        if rm_bg:
            for row in processed_images:
                processed_images_flattened += row

            if len(processed_images_flattened) == 1:
                processed.images = processed_images_flattened
            else:
                processed.images = [images.image_grid(processed_images_flattened,
                                                      rows=p.batch_size * p.n_iter)] + processed_images_flattened

        need_add_watermark_images = []
        need_add_watermark_images1 = []

        new_images = []

        # 添加文字水印之后的操作
        if text_watermark:
            watermarked_images = add_watermark(need_add_watermark_images, need_add_watermark_images1, new_images,
                                               or_images,
                                               text_watermark_color, text_watermark_content, text_watermark_pos,
                                               text_watermark_target,
                                               text_watermark_size, text_watermark_font, custom_font, text_font_path, p,
                                               processed)
            # 添加水印后，只对最终图片进行展示
            processed_images_flattened = []
            for row in watermarked_images:
                processed_images_flattened += row
            if len(processed_images_flattened) == 1:
                processed.images = processed_images_flattened
            else:
                processed.images = [images.image_grid(processed_images_flattened,
                                                      rows=p.batch_size * p.n_iter)] + processed_images_flattened

        # 需要优化逻辑,功能暂时不开放
        if make_a_gif:
            jumps = int(jump)
            dura = dura * jumps
            # 调整输出 GIF 的帧速率
            if frame_rate > 0:
                dura = int(1000 / frame_rate)
            # 反转 GIF 的帧顺序
            if reverse_gif:
                processed_images2 = processed_images2[::-1]

            (fullfn, _) = images.save_image(processed.images[0], p.outpath_grids, "grid",
                                            prompt=p.prompt_for_display, seed=processed.seed, grid=True, p=p)
            for i, row in enumerate(processed_images2):
                fullfn = fullfn[:fullfn.rfind(".")] + "_" + str(i) + ".gif"
                processed_images2[i][0].save(fullfn, save_all=True, append_images=processed_images2[i][1:],
                                             optimize=False, duration=dura, loop=0, disposal=2, fps=frame_rate)

        return processed
