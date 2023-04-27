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

    for prompt_list in prompt_lists:
        if isinstance(prompt_list[0], list):
            combined_prompts = []
            for sub_list in prompt_list:
                combined_prompts.append(random.choice(sub_list))
            selected_prompts.append(", ".join(combined_prompts))
        else:
            selected_prompts.append(random.choice(prompt_list))

    return ", ".join(selected_prompts)


def mcprocess(p, images_num, cb_bi, cb_uw, sd_wt, scene1, scene2, scene3, scene4, scene5):
    prompt_txt = ""
    first_processed = None
    original_images = []
    for i in range(p.batch_size * p.n_iter):
        original_images.append([])

    p.do_not_save_grid = True

    state.job_count = int(images_num * p.n_iter)

    j = -1
    file_idx = 0
    frame_count = 0

    copy_p = copy.copy(p)

    lora_prompts = ['lora:cuteGirlMix4_v10', 'lora:koreandolllikenessV20_v20', 'lora:taiwanDollLikeness_v10',
                    'lora:japanesedolllikenessV1_v15']

    special_index = lora_prompts.index('lora:cuteGirlMix4_v10')
    lora_weights = random_weights(len(lora_prompts), special_index=special_index, special_min=0.4, special_max=0.6)

    combined_lora_prompts_string = ", ".join([f"<{prompt}:{weight}>" for prompt, weight in zip(lora_prompts,
                                                                                               lora_weights)])
    action_prompts = ['yokozuwari', 'ahirusuwari',
                      'indian style', 'kneeling',
                      'arched back', 'lap pillow', 'paw pose',
                      'one knee', 'fetal position', 'on back',
                      'on stomach', 'sitting', 'hugging own legs',
                      'upright straddle', 'standing', 'squatting',
                      'turning around', 'head tilt', 'leaning forward',
                      'arms behind head', 'arms behind back', 'hand in pocket',
                      'cat pose', 'looking afar', 'looking at phone', 'looking away', 'visible through hair',
                      'looking over glasses', 'look at viewer',
                      'close to viewer', 'dynamic angle',
                      'dramatic angle', 'stare', 'looking up', 'looking down', 'looking to the side', 'looking away']

    eyes_prompts = ['one eye closed', 'half-closed eyes', 'aqua eyes', 'glowing eyes', 'pupils sparkling',
                    'color contact lenses', 'long eyelashes', 'colored eyelashes', 'mole under eye', 'lipstick',
                    'heart-shaped mouth', 'pout', 'open mouth', 'closed mouth', ':p', 'chestnut mouth']

    expression_prompts = ['blush stickers', 'blush', 'naughty face', 'light smile', 'seductive smile',
                          'grin', ':d', 'laughing']
    hair_prompts = [
        ['short hair', 'medium hair', 'long hair', 'hair over', 'shoulder'],
        ['white hair', 'blonde hair', 'silver hair', 'grey hair', 'brown hair', 'black hair', 'purple hair', 'red hair',
         'blue hair', 'green hair', 'pink hair', 'orange hair', 'streaked hair', 'multicolored hair',
         'rainbow-like hair'],
        ['bangs', 'crossed bang', 'hair between eye', 'blunt bangs', 'diagonal bangs', 'asymmetrical bangs',
         'braided bangs'],
        ['short ponytail, side ponytail', 'front ponytail', 'split ponytail', 'low twintails', 'short twintails',
         'side braid',
         'braid', 'twin braids', 'ponytail', 'braided ponytail', 'french braid', 'twists', 'high ponytail'],
        ['tied hair', 'single side bun', 'curly hair', 'straight hair', 'wavy hair', 'bob hair', 'slicked-back',
         'pompadour', 'ahoge',
         'antenna hair',
         'heart ahoge', 'drill hair', 'hair wings', 'disheveled hair', 'messy hair', 'chignon', 'braided bun',
         'hime_cut', 'bob cut', 'updo', 'dreadlocks', 'double bun', 'buzz cut', 'big hair', 'shiny hair',
         'glowing hair', 'hair between eyes', 'hair behind ear']
    ]

    hair_accessories_prompts = ['hair ribbon', 'head scarf', 'hair bow', 'crescent hair ornament', 'lolita hairband',
                                'feather hair ornament', 'hair flower', 'hair bun', 'hairclip', 'hair scrunchie',
                                'hair rings',
                                'hair ornament', 'hair stick', 'heart hair ornament']
    jewelry_prompts = ['bracelet', 'ring', 'wristband', 'pendant', 'hoop earrings', 'bangle', 'stud earrings',
                       'sunburst',
                       'pearl bracelet', 'drop earrings', 'puppet rings', 'corsage', 'sapphire brooch', 'jewelry',
                       'necklace', 'brooch']

    camera_perspective_prompts = ['(full body:1.4)', '(medium shot:1.4)']

    default_prompt = ''
    if cb_bi:
        default_prompt = f"(bikini:{sd_wt}), "
    if cb_uw:
        default_prompt = default_prompt + f"(underwear:{sd_wt}), "

    default_prompt = default_prompt + "(8k, best quality, masterpiece:1.2), (realistic, " \
                                      "photo-realistic:1.37), " \
                                      "(solo:2), unity, an extremely delicate and beautiful, extremely detailed, " \
                                      "Amazing, finely detail, masterpiece, best quality, official art, " \
                                      "absurdres, incredibly absurdres, huge filesize, ultra-detailed, " \
                                      "highres, extremely detailed, beautiful detailed girl, extremely detailed " \
                                      "eyes and face, beautiful detailed eyes, Kpop idol, mix4, portrait, " \
                                      "highly detailed skin, no watermark signature, detailed background, " \
                                      "photon mapping," \
                                      "radiosity, physically-based rendering, extremely beautiful, cure lovely, "

    negative_prompt = 'NSFW, Paintings, sketches, (more than one face), (worst quality:2), (low quality:2), ' \
                      '(normal quality:2), bad-picture-chill-75v, ' \
                      '(deformed iris, deformed pupils, bad eyes, semi-realistic:1.4), (bad-image-v2-39000, ' \
                      'bad_prompt_version2, bad-hands-5, EasyNegative, ng_deepnegative_v1_75t, ' \
                      'bad-artist-anime:0.7), ' \
                      '(worst quality, low quality:1.3), (depth of field, blurry:1.2), (greyscale,' \
                      ' monochrome:1.1), ' \
                      'nose, cropped, lowres, text, jpeg artifacts, signature, watermark, username, ' \
                      'blurry, ' \
                      'artist name, trademark, watermark, title, (tan, muscular, loli, petite, child,' \
                      ' infant, ' \
                      'toddlers, chibi, sd character:1.1), multiple view, Reference sheet, long neck,' \
                      ' lowers, ' \
                      'normal quality, ((monochrome)), ((grayscales)), skin spots, acnes, skin blemishes, ' \
                      'age spot, glans, (6 more fingers on one hand), (deformity), multiple breasts, ' \
                      '(mutated hands and fingers:1.5 ), (long body :1.3), (mutation, poorly drawn :1.2), ' \
                      'bad anatomy, liquid body, liquid tongue, disfigured, malformed, mutated, anatomical' \
                      ' nonsense,' \
                      ' text font ui, error, malformed hands, long neck, blurred, lowers, lowres,' \
                      ' bad anatomy,' \
                      ' bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, ' \
                      'bad breasts,' \
                      ' huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, ' \
                      'missing breasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, ' \
                      'missing hand, ' \
                      'disappearing arms, disappearing thigh, disappearing calf, disappearing legs, ' \
                      'fused ears, ' \
                      'bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, ' \
                      'fused animal ears, bad animal ears, poorly drawn animal ears, extra animal ears, ' \
                      'liquid animal ears, heavy animal ears, missing animal ears, text, ui, error, ' \
                      'missing fingers,' \
                      ' missing limb, fused fingers, one hand with more than 5 fingers, ' \
                      'one hand with less than 5 fingers, one hand with more than 5 digit, ' \
                      'one hand with less than 5 digit, extra digit, fewer digits, fused digit, ' \
                      'missing digit, bad digit, liquid digit, colorful tongue, black tongue, cropped,' \
                      ' watermark, username, blurry, JPEG artifacts, signature, 3D, 3D game, 3D game scene,' \
                      ' 3D character, malformed feet, extra feet, bad feet, poorly drawn feet, fused feet, ' \
                      'missing feet, extra shoes, bad shoes, fused shoes, more than two shoes, ' \
                      'poorly drawn shoes, ' \
                      'bad gloves, poorly drawn gloves, fused gloves, bad cum, poorly drawn cum, ' \
                      'fused cum, ' \
                      'bad hairs, poorly drawn hairs, fused hairs, big muscles, ugly, bad face, ' \
                      'fused face,' \
                      ' poorly drawn face, cloned face, big face, long face, bad eyes, ' \
                      'fused eyes poorly drawn eyes,' \
                      ' extra eyes, malformed limbs, more than 2 nipples, missing nipples,' \
                      ' different nipples, ' \
                      'fused nipples, bad nipples, poorly drawn nipples, black nipples, ' \
                      'colorful nipples, gross proportions. short arm, (((missing arms))), missing thighs, ' \
                      'missing calf, missing legs, mutation, duplicate, morbid, mutilated, ' \
                      'poorly drawn hands, ' \
                      'more than 1 left hand, more than 1 right hand, deformed, (blurry), ' \
                      'disfigured, missing legs, ' \
                      'extra arms, extra thighs, more than 2 thighs, extra calf, fused calf, ' \
                      'extra legs, bad knee, ' \
                      'extra knee, more than 2 legs, bad tails, bad mouth, fused mouth, poorly drawn mouth, ' \
                      'bad tongue, tongue within mouth, too long tongue, black tongue, big mouth, ' \
                      ' mouth,' \
                      ' bad mouth, dirty face, dirty teeth, dirty pantie, fused pantie, ' \
                      'poorly drawn pantie, ' \
                      'fused cloth, poorly drawn cloth, bad pantie, yellow teeth, thick lips, bad cameltoe, ' \
                      'colorful cameltoe, bad asshole, poorly drawn asshole, fused asshole, ' \
                      'missing asshole, ' \
                      'bad anus, bad pussy, bad crotch, bad crotch seam, fused anus, fused pussy, ' \
                      'fused anus, ' \
                      'fused crotch, poorly drawn crotch, fused seam, poorly drawn anus, ' \
                      'poorly drawn pussy, ' \
                      'poorly drawn crotch, poorly drawn crotch seam, bad thigh gap, missing thigh gap, ' \
                      'fused thigh gap, liquid thigh gap, poorly drawn thigh gap, poorly drawn anus, ' \
                      'bad collarbone, fused collarbone, missing collarbone, liquid collarbone, ' \
                      'strong girl, ' \
                      'obesity, worst quality, low quality, normal quality, liquid tentacles, ' \
                      'bad tentacles, ' \
                      'poorly drawn tentacles, split tentacles, fused tentacles, missing clit, ' \
                      'bad clit, fused clit,' \
                      ' colorful clit, black clit, liquid clit, QR code, bar code, censored, ' \
                      'safety panties, ' \
                      'safety knickers, beard, furry, pony, pubic hair, mosaic, excrement, faeces, shit'

    for num in range(images_num):
        scene = ''

        if num < images_num / 5:
            scene = scene1
        elif num < (images_num / 5) * 2:
            scene = scene2
        elif num < (images_num / 5) * 3:
            scene = scene3
        elif num < (images_num / 5) * 4:
            scene = scene4
        elif num < images_num:
            scene = scene5

        if 'skirtlift' in scene:
            default_prompt = '(clothes lift), (skirt lift:2), (lifted by self:1.5), (underware:2), skirtlift,' \
                             '<lora:skirtliftTheAstonishing_skirtliftv1:1>, ' + default_prompt
        if state.interrupted:
            state.nextjob()
            break
        if state.skipped:
            state.skipped = False
        state.job = f"{state.job_no + 1} out of {state.job_count}"

        other_prompts = random_prompt_selection([
            action_prompts, eyes_prompts, expression_prompts, hair_prompts,
            hair_accessories_prompts, jewelry_prompts, camera_perspective_prompts,
        ])
        other_prompts = other_prompts + ", " + combined_lora_prompts_string

        if scene != "":
            copy_p.prompt = f"{scene}, {default_prompt}, {other_prompts}"
        else:
            copy_p.prompt = f"{default_prompt}, {other_prompts}"
        copy_p.negative_prompt = negative_prompt

        # 这里按照客户需求固定了参数
        copy_p.width = 540
        copy_p.height = 960
        # copy_p.restore_faces = True
        copy_p.cfg_scale = 7
        copy_p.sampler_name = 'DPM++ SDE Karras'
        copy_p.steps = 20

        processed = process_images(copy_p)
        if first_processed is None:
            first_processed = processed

        for i, img1 in enumerate(processed.images):
            if i > 0:
                break
            original_images[i].append(img1)
        frame_count += 1

    return original_images, first_processed


class Script(scripts.Script):

    def title(self):
        return "随机真人图片生成器"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        with gr.Accordion(label="随机做点图看看吧", open=True):
            with gr.Column():
                images_num = gr.Number(label="请输入要作图的数量", value=0, min=0)
                with gr.Row():
                    cb_bi = gr.Checkbox(label='bikini(比基尼)')
                    cb_uw = gr.Checkbox(label='underwear(内衣)')
                sd_wt = gr.Slider(label='权重', minimum=1, maximum=2)
                scene1 = gr.Textbox(label="请输入你想要的内容1，格式为(内容:2)", value='', lines=1, max_lines=2)
                scene2 = gr.Textbox(label="请输入你想要的内容2，格式为(内容:2)", value='', lines=1, max_lines=2)
                scene3 = gr.Textbox(label="请输入你想要的内容3，格式为(内容:2)", value='', lines=1, max_lines=2)
                scene4 = gr.Textbox(label="请输入你想要的内容4，格式为(内容:2)", value='', lines=1, max_lines=2)
                scene5 = gr.Textbox(label="请输入你想要的内容5，格式为(内容:2)", value='', lines=1, max_lines=2)
                info = gr.HTML("<br>声明：！！！本脚本只提供随机批量作图功能，使用者做的图与脚本作者本人无关！！！")
        return [images_num, cb_bi, cb_uw, sd_wt, scene1, scene2, scene3, scene4, scene5, info]

    def run(self, p, images_num, cb_bi, cb_uw, sd_wt, scene1, scene2, scene3, scene4, scene5, info):

        if p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        p.do_not_save_grid = True

        p.batch_size = 1
        p.n_iter = 1
        original_images, processed = mcprocess(p, int(images_num), cb_bi, cb_uw, float(sd_wt), scene1, scene2, scene3,
                                               scene4,
                                               scene5)

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
