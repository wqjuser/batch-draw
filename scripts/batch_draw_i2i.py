import copy
import importlib.util
import os
import random
import shlex
import subprocess
import sys
import time
import traceback
import gradio as gr
from PIL import Image, ImageSequence, ImageDraw, ImageFont
import modules.scripts as scripts
from modules import images
from modules.processing import process_images
from modules.shared import state
from datetime import datetime


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
batch_draw_i2i_images_folder = "outputs/batch_draw_i2i/images"
batch_draw_i2i_gifs_folder = "outputs/batch_draw_i2i/gifs"
os.makedirs(batch_draw_i2i_images_folder, exist_ok=True)
os.makedirs(batch_draw_i2i_gifs_folder, exist_ok=True)


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


def video2gif(input_video, frames):
    if is_installed('imageio') and is_installed('av'):
        import imageio
        import av

    input_file = input_video
    current_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    output_file = f"{batch_draw_i2i_gifs_folder}/{current_time}.gif"
    inf = input_file.replace("\\", "/")
    inf = inf.replace('"', '')
    print("开始将视频转换为gif请稍后...")
    container = av.open(inf)
    duration = 1 / frames
    with imageio.get_reader(inf, 'ffmpeg') as reader:
        fps = reader.get_meta_data()['fps']
        with imageio.get_writer(output_file, mode='I', duration=duration) as writer:
            for i, frame in enumerate(reader):
                writer.append_data(frame)
    out_path = os.getcwd() + '/' + output_file
    container.close()
    reader.close()
    print("视频转换gif完成，继续后续步骤")
    return out_path


# Count the number of target files
def count_target_files_number(folder_path, suffix):
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(suffix):
            count += 1
    return count


# Counting the number of folders in a folder
def count_subdirectories(path):
    count = 0
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            count += 1
    return count


def get_gif_frame_count(file_path):
    image = Image.open(file_path)
    return image.n_frames


def get_video_frame_count(file_path, fps):
    if is_installed('moviepy'):
        from moviepy.editor import VideoFileClip

    video = VideoFileClip(file_path)
    return int(video.duration * fps)


def get_gif_total_frame_count(directory):
    total_frames = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.gif'):
                file_path = os.path.join(root, file)
                total_frames += get_gif_frame_count(file_path)
    return total_frames


def get_video_total_frame_count(directory):
    total_frames = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp4'):
                file_path = os.path.join(root, file)
                total_frames += get_video_frame_count(file_path)
    return total_frames


# All the image processing is done in this method
def process(p, prompt_txt, file_txt, jump, use_individual_prompts, prompts_folder, max_frames, rm_bg, resize_input,
            resize_dir, width_input, height_input, resize_output, width_output, height_output, mp4_frames, add_bg, bg_path, custom_font,
            text_font_path, text_watermark, text_watermark_color, text_watermark_content,
            text_watermark_font, text_watermark_pos, text_watermark_size, text_watermark_target, save_or):
    # First get the number of all tasks, if there are many files, this process will be time-consuming
    is_single = True
    jumps = int(jump)
    inf = file_txt.replace("\\", "/")
    inf = inf.replace('"', '')
    # If the input address is a single file
    if os.path.isfile(file_txt):
        if inf == "":
            raise ValueError("请输入要使用的视频或者gif图片地址")
        if inf.endswith("mp4"):
            if mp4_frames > 0:
                video_frame_counts = get_video_frame_count(inf, mp4_frames)
                state.job_count = min(int(video_frame_counts * p.n_iter / jumps), max_frames)
            else:
                raise ValueError("请在功能5里面输入大于0的整数帧数")
        elif inf.endswith("gif"):
            gif_frame_counts = get_gif_frame_count(inf)
            state.job_count = min(int(gif_frame_counts * p.n_iter / jumps), max_frames)

        frames = []
        filenames = []
        dura, first_processed, original_images, processed_images, \
            processed_images2, frames_num, filename = deal_with_single_image(file_txt, height_input, jump, max_frames, mp4_frames, p, prompt_txt,
                                                                             prompts_folder, resize_dir, resize_input, rm_bg, use_individual_prompts,
                                                                             width_input, jumps)
        frames.append(frames_num)
        filenames.append(filename)
        images_post_processing(add_bg, bg_path, custom_font, filenames, frames, original_images, p, first_processed, processed_images,
                               processed_images2, rm_bg, save_or, text_font_path, text_watermark, text_watermark_color, text_watermark_content,
                               text_watermark_font, text_watermark_pos, text_watermark_size, text_watermark_target)
        return original_images, first_processed, processed_images, processed_images2, dura, is_single, frames, filenames
    # If the address entered is a folder
    if os.path.isdir(file_txt):
        total_gif_frames = get_gif_total_frame_count(inf)
        total_video_frames = get_video_total_frame_count(inf)
        total_jobs = total_gif_frames + total_video_frames
        state.job_count = min(int(total_jobs * p.n_iter / jumps), max_frames)
        gif_count = count_target_files_number(file_txt, "gif")
        mp4_count = count_target_files_number(file_txt, "mp4")
        total = gif_count + mp4_count
        dura = 0
        frames = []
        results = []
        processed_list = []
        # First determine if the cues are enabled for each frame, and if so, determine if the number of folders
        # in the cue folder matches the number of target files in the input address
        if use_individual_prompts:
            prompts_folders = count_subdirectories(prompts_folder)
            if total != prompts_folders:
                raise ValueError("要添加的提示词文件夹个数与要处理的文件个数不匹配")
            else:
                for file_name in os.listdir(file_txt):
                    file_path = os.path.join(file_txt, file_name)
                    if os.path.isfile(file_path):
                        abs_path = os.path.abspath(file_path)
                        base_name = os.path.basename(abs_path)
                        name_without_extension = os.path.splitext(base_name)[0]
                        hints_subfolder = os.path.join(prompts_folder, name_without_extension)
                        if os.path.isdir(hints_subfolder):
                            result = deal_with_single_image(abs_path, height_input, jump, max_frames, mp4_frames, p, prompt_txt, hints_subfolder,
                                                            resize_dir, resize_input, rm_bg, use_individual_prompts, width_input, jumps)
                            results.append(result)
        else:
            for file_name in os.listdir(file_txt):
                file_path = os.path.join(file_txt, file_name)
                if os.path.isfile(file_path):
                    filenames = []
                    abs_path = os.path.abspath(file_path)
                    result = dura, first_processed, original_images, processed_images, \
                        processed_images2, frames_num, filename = deal_with_single_image(abs_path, height_input, jump, max_frames, mp4_frames, p,
                                                                                         prompt_txt, prompts_folder,
                                                                                         resize_dir, resize_input, rm_bg, use_individual_prompts,
                                                                                         width_input, jumps)
                    frames.append(frames_num)
                    filenames.append(filename)
                    images_post_processing(add_bg, bg_path, custom_font, filenames, frames, original_images, p, first_processed, processed_images,
                                           processed_images2, rm_bg, save_or, text_font_path, text_watermark, text_watermark_color,
                                           text_watermark_content,
                                           text_watermark_font, text_watermark_pos, text_watermark_size, text_watermark_target)
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


def deal_with_single_image(file_txt, height_input, jump, max_frames, mp4_frames, p, prompt_txt, prompts_folder, resize_dir, resize_input, rm_bg,
                           use_individual_prompts, width_input, jumps):
    inf = file_txt.replace("\\", "/")
    inf = inf.replace('"', '')
    file_name = get_file_name_without_extension(inf)
    try:
        if inf == "":
            raise ValueError("请输入要使用的视频或者gif图片地址")
        if inf.endswith("mp4"):
            if mp4_frames > 0:
                if is_installed("moviepy"):
                    inf = video2gif(file_txt, mp4_frames)
            else:
                raise ValueError("请在功能5里面输入大于0的整数帧数")

        if resize_dir != "":
            resize_dir = resize_dir.replace("\\", "/")
            resize_dir = resize_dir.replace('"', '')
        else:
            resize_dir = inf
    except TypeError as e:
        print(e)
    finally:
        gif = Image.open(inf)
        dura = gif.info['duration']

        if rm_bg:
            if is_installed("rembg"):
                import rembg

        # User selection to modify the gif size
        if resize_input:
            if width_input > 0 and height_input > 0:
                gif = Image.open(resize_gif(inf, resize_dir, width_input, height_input))
            else:
                raise ValueError("宽度和高度应该都大于0")

        imgs = [f.copy() for f in ImageSequence.Iterator(gif)]

        if use_individual_prompts:
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

        j = -1
        file_idx = 0
        frame_count = 0

        copy_p = copy.copy(p)
        for img in imgs:
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
            if use_individual_prompts:
                if file_idx < len(prompt_files):
                    prompt_file = os.path.join(prompts_folder, prompt_files[file_idx])
                    # 打开文件，获取文件编码
                    with open(prompt_file, "rb") as f:
                        if is_installed('chardet'):
                            import chardet
                        result = chardet.detect(f.read())
                        file_encoding = result['encoding']
                    with open(prompt_file, "r", encoding=file_encoding) as f:
                        individual_prompt = f.read().strip()
                    copy_p.prompt = f"{individual_prompt}, {copy_p.prompt}"
                    file_idx += 1
                else:
                    print(f"Warning: 输入的提示词文件数量不足,后续图片生成将只使用默认提示词.")

            copy_p.init_images = [img]
            processed = process_images(copy_p)
            if first_processed is None:
                first_processed = processed

            for i, img1 in enumerate(processed.images):
                if i > 0:
                    break
                original_images[i].append(img1)
                if rm_bg:
                    img1 = rembg.remove(img1)
                    processed_images[i].append(img1)
                    img2 = Image.new("RGBA", img1.size, (0, 0, 0, 0))
                    r, g, b, a = img1.split()
                    img2.paste(img1, (0, 0), mask=a)
                    processed_images2[i].append(img2)
                else:
                    processed_images[i].append(img1)
                    processed_images2[i].append(img1)
            frame_count += 1
    return dura, first_processed, original_images, processed_images, processed_images2, frame_count, file_name


def resize_gif(input_path, output_path, width, height):
    with Image.open(input_path) as im:
        frames = []
        # resize each frame of the gif
        for frame in range(im.n_frames):
            im.seek(frame)
            resized = im.resize((width, height), resample=Image.LANCZOS)
            frames.append(resized.copy())

        # save the resized frames as a new gif
        frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=im.info['duration'], loop=0)

    return output_path


# Add a user-specified background image to the image
def add_background_image(foreground_path, background_path, p, processed, filename):
    images_dir = f"{batch_draw_i2i_images_folder}/{formatted_date}"
    # 获取当前工作目录
    cwd = os.getcwd()  # 返回字符串类型
    foreground_path = cwd + "/" + foreground_path[0]
    foreground_path = foreground_path.replace("\\", "/")
    foreground_path = foreground_path.replace('"', '')
    foreground = Image.open(foreground_path).convert('RGBA')
    background = Image.open(background_path).convert('RGBA')
    # 调整背景图片的尺寸以适应前景图片
    background = background.resize(foreground.size)
    # 合并前景和背景图片
    combined = Image.alpha_composite(background, foreground)
    # 保存结果图片
    with_bg_image = images.save_image(combined, f"{images_dir}/{filename}/add_bg_images", "",
                                      prompt=p.prompt_for_display, seed=processed.seed, grid=False, p=p, save_to_dirs=False)
    return with_bg_image


# Add text watermark
def add_watermark(need_add_watermark_images, need_add_watermark_images1, new_images, or_images,
                  text_watermark_color, text_watermark_content, text_watermark_pos, text_watermark_target,
                  text_watermark_size, text_watermark_font, custom_font, text_font_path, p, processed, filenames, frames):
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
                (fullfn, _) = images.save_image(img, f"{batch_draw_i2i_images_folder}/{formatted_date}/{filename}/watermarked_images", "tmp",
                                                prompt=p.prompt_for_display, seed=processed.seed, grid=False, p=p, save_to_dirs=False)
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
                (fullfn, _) = images.save_image(img, f"{batch_draw_i2i_images_folder}/{formatted_date}/{filename}/watermarked_images", "tmp",
                                                prompt=p.prompt_for_display, seed=processed.seed, grid=False, p=p, save_to_dirs=False)
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

    # 删除临时图片
    for path in tmp_images:
        os.remove(path)
    for path in tmp_images1:
        os.remove(path)
    for i, img in enumerate(watered_images):
        watermarked_images[i].append(img)
    return watermarked_images


# Transparent background and add a specified background
def remove_bg(add_bg, bg_path, p, processed, processed_images2, rm_bg, filenames):
    new_images = []
    final_images = []
    images_dir = f"{batch_draw_i2i_images_folder}/{formatted_date}"
    if rm_bg:
        for i, filename in enumerate(filenames):
            for img in processed_images2[i]:
                new_image = images.save_image(img, f"{images_dir}/{filename}/rm_bg_images", "",
                                              prompt=p.prompt_for_display, seed=processed.seed, grid=False, p=p, save_to_dirs=False)
                new_images.append(new_image)
                if add_bg:
                    if bg_path == "":
                        print("未找到要使用的背景图片,将不会添加背景图")
                    else:
                        bg_path = bg_path.replace("\\", "/")
                        bg_path = bg_path.replace('"', '')
                        img1 = add_background_image(new_image, bg_path, p, processed, filename)
                        final_images.append(img1)

        if not add_bg:
            final_images = new_images

    return final_images


# Image post-processing
def images_post_processing(add_bg, bg_path, custom_font, filenames, frames, original_images, p, processed, processed_images,
                           processed_images2, rm_bg, save_or, text_font_path, text_watermark, text_watermark_color, text_watermark_content,
                           text_watermark_font, text_watermark_pos, text_watermark_size, text_watermark_target):
    p.prompt_for_display = processed.prompt
    processed_images_flattened = []
    # here starts the custom image saving logic
    if save_or:
        for i, filename in enumerate(filenames):
            for img in original_images[i]:
                images_dir = f"{batch_draw_i2i_images_folder}/{formatted_date}/{filename}/original_images"
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)
                images.save_image(img, images_dir, "",
                                  prompt=p.prompt_for_display, seed=processed.seed, grid=False, p=p, save_to_dirs=False)
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
    new_images = remove_bg(add_bg, bg_path, p, processed, processed_images2, rm_bg, filenames)
    # Operation after adding a text watermark
    if text_watermark:
        watermarked_images = add_watermark(need_add_watermark_images, need_add_watermark_images1, new_images,
                                           or_images, text_watermark_color, text_watermark_content, text_watermark_pos,
                                           text_watermark_target, text_watermark_size, text_watermark_font, custom_font,
                                           text_font_path, p, processed, filenames, frames)
        # After adding the watermark, only the final image will be displayed
        processed_images_flattened = []
        for row in watermarked_images:
            processed_images_flattened += row
        if len(processed_images_flattened) == 1:
            processed.images = processed_images_flattened
        else:
            processed.images = [images.image_grid(processed_images_flattened,
                                                  rows=p.batch_size * p.n_iter)] + processed_images_flattened


class Script(scripts.Script):

    def title(self):
        return "批量图生图"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        gr.HTML("此脚本可以与controlnet一起使用，若一起使用请把controlnet的参考图留空。")
        with gr.Accordion(label="基础属性，必填项，每一项都不能为空", open=True):
            with gr.Column(variant='panel'):
                jump = gr.Dropdown(["1", "2", "3", "4", "5"], label="1. 跳帧(1为不跳，2为两帧取一......)", value="1")
                prompt_txt = gr.Textbox(label="2. 默认提示词，将影响各帧", lines=3, max_lines=5, value="")
                file_txt = gr.Textbox(
                    label="3. 输入gif或mp4文件或文件夹全路径(勿出现中文), 若为mp4请勾选功能5并设置帧数默认30帧每秒, 处理mp4会很耗时",
                    lines=1,
                    max_lines=5,
                    value="")
                max_frames = gr.Number(
                    label="4. 输入指定的 GIF 帧数 (运行到指定帧数后停止(不会大于GIF的总帧数)，用于用户测试生成图片的效果，最小1帧,测试完成后请输入一个很大的数保证能把GIF所有帧数操作完毕)",
                    value=666,
                    min=1
                )
        with gr.Accordion(label="附加选项，根据需要使用", open=False):
            with gr.Column(variant='panel'):
                mp4togif = gr.Checkbox(
                    label="5. 启用mp4转gif,只有输入为mp4文件时勾选起效，启用视频处理会安装一些必要的仓库，首次安装会耗时，如果安装失败请手动安装")

                def check_moviepy(mp4togif):
                    if mp4togif:
                        if is_installed('imageio') and is_installed('av') and is_installed('moviepy'):
                            return gr.update(visible=False, value='')
                        else:
                            return gr.update(visible=True, value='请点击安装环境')
                    else:
                        return gr.update(visible=False, value='')

                btn_install_moviepy = gr.Button(value="未安装处理视频的环境，请点击安装", visible=False).style(
                    full_width=True)
                mp4togif.change(check_moviepy, inputs=[mp4togif], outputs=[btn_install_moviepy])

                def install_moviepy():
                    if is_installed('imageio') and is_installed('av') and is_installed('moviepy'):
                        return
                    print("安装过程：", run_pip("install imageio av moviepy", "开始安装环境"))
                    return gr.update(value="安装完成", visible=False)

                def change_btn_ui():
                    return gr.update(value='安装中,请在控制台查看进度,安装完成按钮将会自动消失', interactive=False)

                # 添加点击事件
                btn_install_moviepy.click(change_btn_ui, inputs=None, outputs=[btn_install_moviepy],
                                          show_progress=True)
                btn_install_moviepy.click(install_moviepy, inputs=None, outputs=[btn_install_moviepy],
                                          show_progress=True)
                mp4_frames = gr.Number(
                    label="输入指定的视频转gif帧数",
                    value=30,
                    min=1
                )
                use_individual_prompts = gr.Checkbox(
                    label="6. 为每一帧选择一个提示词文件（非必选） 可以丰富人物表情，但是由于每一张的tag不同会引起视频的人物脸部不太统一",
                    value=False
                )
                prompts_folder = gr.Textbox(
                    label="输入包含提示词文本文件的文件夹路径(勾选上方单选框起效)",
                    lines=1,
                    max_lines=5,
                    value=""
                )

            with gr.Accordion(label="去除背景和保留原图(至少选择一项否则文件夹中没有保留生成的图片)", open=True):
                with gr.Column(variant='panel'):
                    with gr.Row():
                        rm_bg = gr.Checkbox(label="7. 去除图片的背景仅保留人物?",
                                            info="需要安装rembg，若未安装请点击下方按钮安装rembg")
                        save_or = gr.Checkbox(label="8. 是否保留原图", info="为了不影响查看原图，默认选中会保存未删除背景的图片", value=True)

                    def check_rembg(rm_bg):
                        if rm_bg:
                            if is_installed('rembg'):
                                return gr.update(visible=False, value='')
                            else:
                                return gr.update(visible=True, value='请点击安装rembg')
                        else:
                            return gr.update(visible=False, value='')

                    btn_install_rembg = gr.Button(value="未安装rembg，请点击安装rembg", visible=False).style(
                        full_width=True)
                    rm_bg.change(check_rembg, inputs=[rm_bg], outputs=[btn_install_rembg])

                    def install_rembg():
                        if is_installed("rembg"):
                            return
                        return gr.update(value="安装完成", visible=False)

                    def change_btn_ui():
                        return gr.update(value='安装中,请在控制台查看进度,安装完成按钮将会自动消失', interactive=False)

                    # 添加点击事件
                    btn_install_rembg.click(change_btn_ui, inputs=None, outputs=[btn_install_rembg],
                                            show_progress=True)
                    btn_install_rembg.click(install_rembg, inputs=None, outputs=[btn_install_rembg],
                                            show_progress=True)

                    add_bg = gr.Checkbox(
                        label="为透明图片自定义背景图片",
                        value=False
                    )
                    bg_path = gr.Textbox(
                        label="输入自定义背景图片路径(勾选上方单选框以及功能6起效)",
                        lines=1, max_lines=2,
                        value=""
                    )

            with gr.Accordion(label="更多操作(打开看看说不定有你想要的功能)", open=False):
                with gr.Column(variant='panel'):
                    resize_input = gr.Checkbox(label="9. 调整输入GIF尺寸",
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
                        text_watermark = gr.Checkbox(label="10. 添加文字水印", info="自定义文字水印")
                        with gr.Row():
                            with gr.Column(scale=8):
                                with gr.Row():
                                    text_watermark_font = gr.Dropdown(
                                        ["微软雅黑", "宋体", "黑体", "楷体", "仿宋宋体"],
                                        label="内置5种字体,启用自定义后这里失效",
                                        value="微软雅黑")
                                    text_watermark_target = gr.Dropdown(["0", "1", "2"],
                                                                        label="水印添加对象(0:原始,1:透明,2:全部)",
                                                                        value="0")
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
                                label="输入自定义水印字体路径(勾选左边自定义水印字体单选框以及功能9起效)",
                                lines=1, max_lines=2,
                                value=""
                            )

        return [jump, prompt_txt, file_txt, max_frames, use_individual_prompts, prompts_folder, rm_bg, save_or,
                btn_install_rembg, resize_input, resize_dir, width_input, height_input, resize_output, resize_target,
                width_output, height_output, make_a_gif, frame_rate, reverse_gif, text_watermark, text_watermark_font,
                text_watermark_target, text_watermark_pos, text_watermark_color, text_watermark_size,
                text_watermark_content, custom_font, text_font_path, add_bg, bg_path, mp4_frames]

    def run(self, p, jump, prompt_txt, file_txt, max_frames, use_individual_prompts, prompts_folder, rm_bg, save_or,
            btn_install_rembg, resize_input, resize_dir, width_input, height_input, resize_output, resize_target,
            width_output, height_output, make_a_gif, frame_rate, reverse_gif, text_watermark, text_watermark_font,
            text_watermark_target, text_watermark_pos, text_watermark_color, text_watermark_size,
            text_watermark_content, custom_font, text_font_path, add_bg, bg_path, mp4_frames):

        if p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        p.do_not_save_grid = True
        # here the logic for saving images in the original sd is disabled
        p.do_not_save_samples = True

        p.batch_size = 1
        p.n_iter = 1
        processed = process(
            p, prompt_txt, file_txt, jump, use_individual_prompts, prompts_folder, int(max_frames), rm_bg, resize_input,
            resize_dir, int(width_input), int(height_input), resize_output, int(width_output), int(height_output),
            int(mp4_frames), add_bg, bg_path, custom_font, text_font_path, text_watermark, text_watermark_color, text_watermark_content,
            text_watermark_font, text_watermark_pos, text_watermark_size, text_watermark_target, save_or)

        # Need to optimize the logic, the function is not open for the time being
        if make_a_gif:
            jumps = int(jump)
            dura = dura * jumps
            # Adjust the frame rate of the output GIF
            if frame_rate > 0:
                dura = int(1000 / frame_rate)
            # Reversing the frame order of a GIF
            if reverse_gif:
                processed_images2 = processed_images2[::-1]

            (fullfn, _) = images.save_image(processed.images[0], p.outpath_grids, "grid",
                                            prompt=p.prompt_for_display, seed=processed.seed, grid=True, p=p)
            for i, row in enumerate(processed_images2):
                fullfn = fullfn[:fullfn.rfind(".")] + "_" + str(i) + ".gif"
                processed_images2[i][0].save(fullfn, save_all=True, append_images=processed_images2[i][1:],
                                             optimize=False, duration=dura, loop=0, disposal=2, fps=frame_rate)

        return processed
