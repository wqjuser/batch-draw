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


def random_weights(num_weights, max_sum=1.0):
    weights = [random.random() for _ in range(num_weights - 1)]
    weights.append(0.0)
    weights.append(max_sum)
    weights.sort()
    return [weights[i + 1] - weights[i] for i in range(num_weights)]


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


def mcprocess(p, images_num):
    prompt_txt = ""
    first_processed = None
    original_images = []
    for i in range(p.batch_size * p.n_iter):
        original_images.append([])

    p.do_not_save_grid = True
    job_count = 0
    jobs = []

    lines = [x.strip() for x in prompt_txt.splitlines()]
    lines = [x for x in lines if len(x) > 0]

    for line in lines:
        args = {"prompt": line}
        n_iter = args.get("n_iter", 1)
        job_count += 1
        jobs.append(args)

    state.job_count = int(images_num * p.n_iter)

    j = -1
    file_idx = 0
    frame_count = 0

    copy_p = copy.copy(p)

    lora_prompts = ['lora:koreanDollLikeness_v15', 'lora:taiwanDollLikeness_v10',
                    'lora:japaneseDollLikeness_v10', 'lora:cuteGirlMix4_v10']

    lora_weights = random_weights(len(lora_prompts))

    combined_lora_prompts = []
    for prompt, weight in zip(lora_prompts, lora_weights):
        combined_lora_prompts.append(f"{prompt}:{weight}")

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
                    'color contact lenses', 'long eyelashes', 'colored eyelashes', 'mole under eye']

    expression_prompts = ['blush stickers', 'blush', 'blank stare', 'nervous', 'confused', 'scared', 'light frown',
                          'frown', 'naughty face', 'zzz', 'light smile', 'false smile', 'seductive smile', 'smirk',
                          'seductive smile', 'grin', ':d', 'laughing']
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

    env_prompts = ['bathroom', 'indoor', 'classroom', 'gym']

    camera_perspective_prompts = ['Depth of field', 'Panorama', 'telephoto lens', 'macro lens', 'full body',
                                  'medium shot',
                                  'cowboy shot', 'profile picture', 'close up portrait', 'POV']

    default_prompt = "(lolita fashion style clothes:2), (8k, best quality, masterpiece:1.2), (realistic, " \
                     "photo-realistic:1.37), " \
                     "(solo:2),unity, an extremely delicate and beautiful, extremely detailed, " \
                     "Amazing, finely detail, masterpiece, best quality, official art, " \
                     "absurdres, incredibly absurdres, huge filesize, ultra-detailed, " \
                     "highres, extremely detailed, beautiful detailed girl, extremely detailed " \
                     "eyes and face, beautiful detailed eyes, Kpop idol, mix4, smile, portrait, " \
                     "highly detailed skin, no watermark signature, detailed background, photon mapping," \
                     " radiosity, physically-based rendering, extremely beautiful, cure lovely, "

    for num in images_num:
        if state.interrupted:
            state.nextjob()
            break
        if state.skipped:
            state.skipped = False
        state.job = f"{state.job_no + 1} out of {state.job_count}"

        for k, v in args.items():
            setattr(copy_p, k, v)

        other_prompts = random_prompt_selection([
            action_prompts, eyes_prompts, expression_prompts, hair_prompts,
            hair_accessories_prompts, jewelry_prompts, env_prompts, camera_perspective_prompts
        ])

        copy_p.prompt = f"{default_prompt}, {other_prompts}"

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
        return "随机图片生成器"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):

        with gr.Column():
            images_num = gr.Number(label="请输入要作图的数量", value=0, min=0)
        return [images_num]

    def run(self, p, images_num):

        if p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        p.do_not_save_grid = True

        p.batch_size = 1
        p.n_iter = 1
        original_images, processed = mcprocess(p, int(images_num))

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
