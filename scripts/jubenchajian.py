import copy
import math
import os
import random
import sys
import traceback
import shlex
import re

import modules.scripts as scripts
import gradio as gr

from modules import sd_samplers
from modules.processing import Processed, process_images
from PIL import Image
from modules.shared import opts, cmd_opts, state


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag == "true") else False


prompt_tags = {
    "sd_model": None,  # 翻译过来是 生成对抗模型
    "outpath_samples": process_string_tag,  # 翻译过来是 输出样本路径
    "outpath_grids": process_string_tag,  # 翻译过来是 输出网格路径
    "prompt_for_display": process_string_tag,  # 翻译过来是 显示提示
    "prompt": process_string_tag,  # 翻译过来是 提示
    "negative_prompt": process_string_tag,  # 翻译过来是 负提示
    "styles": process_string_tag,  # 翻译过来是 样式
    "seed": process_int_tag,  # 翻译过来是 种子
    "subseed_strength": process_float_tag,  # 翻译过来是 子种子强度
    "subseed": process_int_tag,  # 翻译过来是 子种子
    "seed_resize_from_h": process_int_tag,  # 翻译过来是 从高度调整种子大小
    "seed_resize_from_w": process_int_tag,  # 翻译过来是 从宽度调整种子大小
    "sampler_index": process_int_tag,  # 翻译过来是 采样器索引
    "sampler_name": process_string_tag,  # 翻译过来是 采样器名称
    "batch_size": process_int_tag,  # 翻译过来是 批量大小
    "n_iter": process_int_tag,  # 翻译过来是 迭代次数
    "steps": process_int_tag,  # 翻译过来是 步数
    "cfg_scale": process_float_tag,  # 翻译过来是 配置比例
    "width": process_int_tag,  # 翻译过来是 宽度
    "height": process_int_tag,  # 翻译过来是 高度
    "restore_faces": process_boolean_tag,  # 翻译过来是 恢复面部
    "tiling": process_boolean_tag,  # 翻译过来是 平铺
    "do_not_save_samples": process_boolean_tag,  # 翻译过来是 不保存样本
    "do_not_save_grid": process_boolean_tag  # 翻译过来是 不保存网格
}

ooo = "Black clothes, (black hair),"


def cmdargs(line):  # 函数功能是  命令行参数
    args = shlex.split(line)
    pos = 0
    res = {}

    while pos < len(args):
        arg = args[pos]

        assert arg.startswith("--"), f'must start with "--": {arg}'  # 翻译过来是 必须以“--”开头
        assert pos + 1 < len(args), f'missing argument for command line option {arg}'  # 翻译过来是 命令行选项缺少参数

        tag = arg[2:]

        if tag == "prompt" or tag == "negative_prompt":  # 翻译过来是 提示或负提示
            pos += 1
            prompt = args[pos]
            pos += 1
            while pos < len(args) and not args[pos].startswith("--"):  # 翻译过来是 以“--”开头
                prompt += " "
                prompt += args[pos]
                pos += 1
            res[tag] = prompt
            continue

        func = prompt_tags.get(tag, None)  # 翻译过来是 提示标签
        assert func, f'unknown commandline option: {arg}'  # 翻译过来是 未知的命令行选项

        val = args[pos + 1]  # 翻译过来是 值
        if tag == "sampler_name":  # 翻译过来是 采样器名称
            val = sd_samplers.samplers_map.get(val.lower(), None)  # 翻译过来是 采样器映射

        res[tag] = func(val)  # 翻译过来是 函数

        pos += 2  # 翻译过来是 位置

    return res  # 上面这个函数的功能是  通过命令行参数来设置参数


mtt1 = "(((Black and white comic page content))),(Black hair)，"
mtt2 = " ((Colorful manga,Dialogue bubble box, Animated movies)),"
mtt3 = "Magnificent and breathtaking wallpapers that steal the show, "


def load_prompt_file(file):
    if file is None:
        lines = []
    else:
        lines = [x.strip() for x in file.decode('utf8', errors='ignore').split("\n")]

    return None, "\n".join(lines), gr.update(lines=7)


def ddd(num):
    if num == 0:
        return ""
    else:
        return '(' * num + 'Sense of speed' + ')' * num


def ttxq(num):
    if num == 0:
        return ""
    else:
        return '(' * num + 'Full-frame fisheye visualization，' + ')' * num


mqfd = "Close-up, extreme close-up, medium close-up, medium shot, medium long shot, long shot, extreme long shot, full shot, cowboy shot, bird's eye view, worm's eye view, high angle, low angle, Dutch angle, straight-on angle, over-the-shoulder shot, point-of-view shot, two-shot, three-shot, establishing shot, cutaway shot, reaction shot, insert shot, off-screen shot, reverse angle, top shot, bottom shot, tilt shot, pan shot, zoom in shot, zoom out shot, dolly in shot, dolly out shot, tracking shot, steadicam shot, handheld shot, crane shot, aerial shot, split screen shot, freeze frame shot."


def qqq(num):
    if num == 0:
        return ""
    elif num < 0:
        return '(' * abs(num) + '(Cute art style),' + ')' * abs(num)
    else:
        return '(' * num + 'Realistic style,' + ')' * num


def tttrt(fff):
    if fff == 0:
        return ""
    elif fff < 0:
        return '(' * abs(fff) + '(Horror, gloomy visuals),' + ')' * abs(fff)
    else:
        return '(' * fff + 'Sunshine, optimistic visuals,' + ')' * fff


def get_name(name_str):
    name_list = name_str.split('-')
    return '({})'.format(name_list[0].strip())


def txttf(text):
    pattern = r'[。！？\?\n；;,\.\s]'
    lines = re.split(pattern, text)
    lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(lines)


def kkkk(sentence, word_str):
    word_dict = {}
    try:
        for pair in word_str.split(","):
            if not pair:
                continue
            old_word, new_word = pair.split("-")
            word_dict[old_word.strip()] = new_word.strip()
    except ValueError:
        return sentence
    for old_word, new_word in word_dict.items():
        if old_word.isupper():
            sentence = sentence.replace(old_word, new_word.upper())
        else:
            sentence = sentence.replace(old_word, new_word)
    result = sentence.title()
    result = re.sub('(<.*?>)', lambda m: m.group().lower(), result)
    return result


qqta = "把剧本填写在这里~~(1 girl)，(1 boy)，(2 people)，"

ppt = ",Movie shots, "
dddf = ",stone, "
mtt = ",kkk, "
cccctt = "Left, right, up, down, forward, backward, north, south, east, west, northeast, northwest, southeast, southwest, horizontal, vertical, diagonal, ascending, descending, clockwise"
ggg = ""
dddtt = "Horizon, Cascade, Serenity, Luminous, Whirlpool, Twilight, Radiance, Oasis, Spectrum, Reflection, Infinity, Aurora, Harmony, Velocity, Enigma, Eclipse, Galaxy, Mirage, Thunderstorm, Cosmos,Synthesis, Blossom, Chaos, Solitude, Vibrance, Illusion, Euphoria, Nebula, Phoenix, Melancholy"
mcc = ""
ddc = ""
qdf = ""


class Script(scripts.Script):
    def title(self):
        return "AI漫画助手v3.0 作者咸蛋酱"

    def ui(self, is_img2img):

        with gr.Row():
            xtf = gr.Dropdown(label="选择作者画风（预设）-- [如果无效说明模型不包含这类数据]",
                              choices=[" - 无", "Loish - 创新者", "Kim Jung Gi - 无限想象", "Artgerm - 超级女孩", "Sakimichan - 魔法奇谭",
                                       "James Paick - 未来科技", "Jock - 末日幸存者", "Bang Sangho - 斗鱼", "Stanley Lau - 正义联盟",
                                       "Yuehui Tang - 神话世界", "Krenz Cushart - 战斗机甲", "WLOP - 粉色噩梦", "Guweiz - 幸存者",
                                       "Fishball - 三体：死神永生", "Loish - 梦境之旅", "Artgerm - 帝国的崛起", "Hoon - 地球之子",
                                       "Kuvshinov Ilya - 暗影猎人", "Redjuice - 遥远的未来", "Koyorin - 星际公主", "Sakimichan - 龙珠：超级赛亚人",
                                       "James Paick - 星际之战", "Sparth (Nicolas Bouvier) - 未来城市", "Craig Mullins - 地球未来",
                                       "Ryan Meinerding - 漫威电影", "John Wallin Liberto - 外星空间", "Fenghua Zhong - 远古王朝",
                                       "Mike Nash - 魔法学院", "Sergey Kolesov - 超现实幻想", "Eytan Zana - 极光奇观", "Simon Weaner - 反乌托邦世界",
                                       "Stanley Lau - 神奇女侠: 马蒂斯之死", "Brom - 黑暗精灵", "Jock - 蝙蝠侠: 黑暗骑士归来", "Oliver Coipel - 金刚",
                                       "Beksinski - 外星人", "Yoshitaka Amano - 尤迪安", "Kekai Kotaki - 龙族起源", "Sachin Teng - 雪之女王",
                                       "Jasmine Becket-Griffith - 魔法国度", "Pascal Blanche - 星际探险家", "Joakim Ericsson - 阿拉德历险记",
                                       "Jesper Ejsing - 龙与地下城", "Ian McQue - 星际迷航", "Ryan Meinerding - 钢铁侠：英雄崛起",
                                       "Kekai Kotaki - 光与暗的传说", "Ruan Jia - 火影忍者", "Bastien Lecouffe Deharme - 神话传说",
                                       "Ryan Lang - 奇幻森林", "Daniel Kamarudin - 《魔兽世界》", "Eytan Zana - 未来战士",
                                       "Stan Lee - 蜘蛛侠、钢铁侠、雷神等漫威英雄", "Todd McFarlane - 脊柱之战、自杀小队", "Neil Gaiman - 沙曼、无人洛城",
                                       "Dave Gibbons - 世纪杀手", "Frank Miller - 蝙蝠侠：黑暗骑士归来、罪恶之城",
                                       "Bill Watterson - 柯南·伊素格、公园和斯皮格", "Osamu Tezuka - 铁臂阿童木、星球大战、菠萝超人",
                                       "Hayao Miyazaki - 龙猫、天空之城、千与千寻", "Robert Crumb - 乌托邦", "Al Hirschfeld - 百老汇剧院的演员插画",
                                       "Daniel Danger - 最后生还者 (插画师)", "Dave Rapoza - 忍者神龟 (插画师)",
                                       "Jessica Hische - 企鹅出版社 (插画师)", "Jock - 蝙蝠侠 (插画师)", "Ken Taylor - 盗梦空间 (插画师)",
                                       "Lauren Hom - 谷歌 (插画师)", "Loish - Wacom (插画师)", "Mike Mitchell - 漫威 (插画师)",
                                       "Shantell Martin - 纽约城市芭蕾舞团 (插画师)", "Stephen Bliss - 侠盗猎车手 (插画师)", "Steve Prescott - 战锤",
                                       "Yoshitaka Amano - 最终幻想", "Akira Toriyama - 龙珠", "Hayao Miyazaki - 千与千寻",
                                       "Osamu Tezuka - 铁臂阿童木", "Masamune Shirow - 攻壳机动队", "Takehiko Inoue - 灌篮高手",
                                       "Katsuhiro Otomo - AKIRA", "Tsutomu Nihei - BLAME!", "Kazuo Umezu - 漂流教室",
                                       "Hirohiko Araki - JoJo的奇妙冒险", "Rei Hiroe - 黑色底盘", "Yusuke Murata - 一拳超人",
                                       "Kentaro Miura - 烙印勇士", "Fumiya Sato - 杀戮都市", "Naoki Urasawa - MONSTER", "Tite Kubo - 死神",
                                       "Hiromu Arakawa - 钢之炼金术师", "Takeshi Obata - 死亡笔记", "CLAMP collective - 魔卡少女樱",
                                       "Rumiko Takahashi - 犬夜叉", "Ralph McQuarrie - 星球大战", "Syd Mead - 洛城机密", "Rick Baker - 狼人、星际迷航",
                                       "H.R. Giger - 异形", "Jean Giraud (Moebius) - 第五元素", "Yoshitaka Amano - 最终幻想",
                                       "Geof Darrow - 黑客帝国", "Nilo Rodis-Jamero - 星球大战", "Doug Chiang - 星球大战、阿凡达", "John Howe - 魔戒",
                                       "Alan Lee - 魔戒、霍比特人", "Stuart Craig - 哈利波特", "Kevin O'Neill - 世纪之战", "Alex McDowell - 迷雾之城",
                                       "Ken Adam - 詹姆斯•邦德系列电影", "CLAMP - 经久纱流、大场水滸传、某个科学的超电磁炮",
                                       "Koyoharu Gotouge - 鬼灭之刃", "Hajime Isayama - 进击的巨人", "Mitsuru Adachi - 双截龙",
                                       "Inio Asano - 国王游戏、好想告诉你", "Yusei Matsui - 暗杀教室", "Junji Ito - Uzumaki、镰仓物语、恶魔之谷",
                                       "Hiro Mashima - FAIRY TAIL、EDENS ZERO", "Kohei Horikoshi - 像素英雄",
                                       "Aka Akasaka - 与谍同谋、被讨厌的松本同学", "Kentarō Miura - 灌篮高手、小说家、新暗行御史",
                                       "Kengo Hanazawa - 吃人鬼、我与僵尸有个约会", "Yuki Tabata - 黑色五叶草", "Gege Akutami - 呪術廻戦",
                                       "Kousuke Oono - 一弹起始、蓝天航线", "Akiko Higashimura - 刺客伍六三、东京女子图鉴、虚构推理",
                                       "Io Sakisaka - 好想告诉你、罗曼蒂克的玩具、Honey and Clover", "Haruko Ichikawa - 一人之下、千古王者、山海情",
                                       "Haruka Kanda - 紫罗兰永恒花园、花开伊吕波、猎魔人", "Miyuki Nakayama - 战斗陀螺、洛克人X",
                                       "Paru Itagaki - 社长，戏说社员的恋爱", " Kouhei Aonishi - 大声展开！伸展吧！我的腰椎、钓鱼之夏、深夜食堂",
                                       "Fujimoto Tatsuki - 鬼滅之刃 炭治郎外传 悲伤之刃、CHAINSAWMAN",
                                       "Yuito Kimura - 重新起航的天之歌、谷底的那一抹阳光、白色相簿2",
                                       "Yuji Kaku - 圣断罗斯之魔女、绝命诗1816、麻衣的宇宙奇幻之旅", "Yoshitaka Amano - 宝石之国（插画师）",
                                       "Range Murata - 最终幻想XII、法师与猫（插画师）", "Kazuki Takahashi - 游戏王（插画师）",
                                       "Akihiko Yoshida - 光之海（插画师）", "Shunya Yamashita - 信长之野望Online、绝对领域战争（插画师）",
                                       "Tony Taka - 无双OROCHI系列、Fault！（插画师）", "Nishikiito - 食戟之灵（插画师）",
                                       "REDJUICE - Guilty Crown、Beatless（插画师）", "WIT STUDIO - 进击的巨人、在下坂本，有何贵干？（动画公司）",
                                       "Hiroshi Nagai - F-1 Grand Prix、F-1 Hero（插画师）", "HACCAN - 武器种族传说、墓之沙（插画师）",
                                       "Kisai Takayama - 魔卡少女樱、魔法骑士雷阿斯（插画师）", "CLAMP - 神之塔（插画师）",
                                       "Yuka Nakajima - Fate/Grand Order、乐园追放（插画师）", "Akiman - 街头霸王（插画师）",
                                       "Shigenori Soejima - 女神异闻录系列、CATHERINE FULL BODY（插画师）", "Koyori - 只有我能进入的隐藏迷宫（插画师）",
                                       "Oyari Ashito - 学院黙示录、英雄伝説 空の軌跡（插画师）", "Sakura Hanpen - 干支魂（插画师）",
                                       "Yoshinori Shizuma - 决斗！平安京、罪人与龙（插画师）"], value="Yoshinori Shizuma - 决斗！平安京、罪人与龙（插画师）",
                              elem_id=self.elem_id("xtf"))
            rtf = gr.Dropdown(label="选择时代背景（预设）",
                              choices=[" - 无", "Middle Ages - 中世纪", "Renaissance - 文艺复兴", "Meiji Period - 日本明治時代",
                                       "Industrial Revolution - 工业革命", "Edo Period - 日本江戸時代", "Roaring Twenties - 繁华的二十年代",
                                       "Cold War era - 冷战时期", "Information Age - 信息时代", "Song Dynasty - 中华宋朝", "Digital Age - 数字时代",
                                       "Warring States Period - 中华战国时期", "Bronze Age - 青铜时代", "Iron Age - 铁器时代",
                                       "Classical Antiquity - 古典时代", "Victorian Era - 维多利亚时代", "Gilded Age - 镀金时代",
                                       "Jazz Age - 爵士时代", "Space Age - 太空时代", "Ancient Egypt - 古埃及",
                                       "Golden Age of Hollywood - 好莱坞黄金时代", "Tang Dynasty - 中华唐朝", "Post-Modernism - 后现代主义",
                                       "Era of Good Feelings - 平和年代", "Age of Enlightenment - 启蒙时代", "Gothic Period - 哥特式时期",
                                       "Age of Exploration - 探险时代", "Art Nouveau - 新艺术运动", "Ming Dynasty - 中华明朝",
                                       "Atomic Age - 原子时代", "Baroque Period - 巴洛克时期", "Modernism - 现代主义"],
                              value=" - 无", elem_id=self.elem_id("rtf"))

            # with gr.Row():
        #      stf = gr.Dropdown(label="选择色调（预设）", choices=[" - 无","Warm - 暖色调", "Cool - 冷色调","Neutral - 中性色调","Earthy - 泥土色调","Pastel - 柔和色调","Vibrant - 鲜艳色调","Muted - 柔和色调","Split-complementary - 分裂互补色调","Monochromatic - 单色调",], value="Muted - 柔和色调", elem_id=self.elem_id("rtf"))
        #     btf = gr.Dropdown(label="选择漫画类型（预设）", choices=[" - 无","Shonen Manga - 少年漫画","Shojo Manga - 少女漫画","Seinen Manga - 青年漫画","Science Fiction Manga - 科幻漫画","Gekiga - 剧情漫畫","Kodomomuke Manga - 儿童漫画","Fantasy Manga - 奇幻漫画","Horror Manga - 恐怖漫画","Sports Manga - 运动漫画", "Yaoi Manga - 耽美漫画","Yuri Manga - 百合漫画",], value="Shonen Manga - 少年漫画", elem_id=self.elem_id("rtf"))

        with gr.Row():
            seedX = gr.Number(label="Seed值,-1是随机，其他任意是稳定角色画风", value=333, precision=2, elem_id="seedX")
            MXT = gr.Number(label="连跑轮次，改多少轮，它就会跑多少轮，适合开随机变化抽卡", value=1, precision=2, elem_id="MXT")

        with gr.Row():
            CX1 = gr.Checkbox(label="随机微调【开启后，每个图的输出，随机细微变化，微调】", value=False, display="inline", elem_id=self.elem_id("CX1"))
            CXt = gr.Checkbox(label="随机镜头【开启后会随机镜头拍摄，开启后镜头更生动！】", value=False, display="inline", elem_id=self.elem_id("CXt"))
        tttr = gr.Slider(minimum=-6, maximum=6, step=1, label='画面气氛【左边恐怖阴暗，右边明亮乐观】', value=0, elem_id=self.elem_id("tttr"))
        CXx = gr.Checkbox(label="分割文本【开启后会自动分割文本，适合粘贴整本小说用的。】", value=False, display="inline", elem_id=self.elem_id("CXx"))
        PTX = gr.Textbox(
            label="列表输入，这里输入批处理文本或者剧本，每行会输出一张图。【推荐使用GPT来写分镜，一段一个分镜】【推荐画布尺寸：300*450，450*300 高清就开放大】",
            value=qqta, lines=1, elem_id=self.elem_id("PTX"))
        with gr.Row():
            M1 = gr.Textbox(label="主角描述1，例如：lala-穿着红色旗袍的女孩", lines=1, elem_id=self.elem_id("M1"))
            M2 = gr.Textbox(label="主角描述2，例如：jack-穿着绿色毛衣的男孩", lines=1, elem_id=self.elem_id("M2"))
            M3 = gr.Textbox(label="主角描述3，例如：maka-身上破烂不堪的僵尸", lines=1, elem_id=self.elem_id("M3"))
            M4 = gr.Textbox(label="主角描述4，例如：sara-穿花衬衫的30岁大叔", lines=1, elem_id=self.elem_id("M4"))
        with gr.Row():
            M5 = gr.Textbox(label="主角描述5，例如：lala-穿着红色旗袍的女孩", lines=1, elem_id=self.elem_id("M5"))
            M6 = gr.Textbox(label="主角描述6，例如：jack-穿着绿色毛衣的男孩", lines=1, elem_id=self.elem_id("M6"))
            M7 = gr.Textbox(label="主角描述7，例如：maka-身上破烂不堪的僵尸", lines=1, elem_id=self.elem_id("M7"))
            M8 = gr.Textbox(label="主角描述8，例如：sara-穿花衬衫的30岁大叔", lines=1, elem_id=self.elem_id("M8"))
        with gr.Row():
            M9 = gr.Textbox(label="主角描述9，例如：lala-穿着红色旗袍的女孩", lines=1, elem_id=self.elem_id("M9"))
            M10 = gr.Textbox(label="主角描述10，例如：jack-穿着绿色毛衣的男孩", lines=1, elem_id=self.elem_id("M10"))
            M11 = gr.Textbox(label="主角描述11，例如：maka-身上破烂不堪的僵尸", lines=1, elem_id=self.elem_id("M11"))
            M12 = gr.Textbox(label="主角描述12，例如：sara-穿花衬衫的30岁大叔", lines=1, elem_id=self.elem_id("M12"))
        with gr.Row():
            M13 = gr.Textbox(label="主角描述13，例如：lala-穿着红色旗袍的女孩", lines=1, elem_id=self.elem_id("M13"))
            M14 = gr.Textbox(label="主角描述14，例如：jack-穿着绿色毛衣的男孩", lines=1, elem_id=self.elem_id("M14"))
            M15 = gr.Textbox(label="主角描述15，例如：maka-身上破烂不堪的僵尸", lines=1, elem_id=self.elem_id("M15"))
            M16 = gr.Textbox(label="主角描述16，例如：sara-穿花衬衫的30岁大叔", lines=1, elem_id=self.elem_id("M16"))
        with gr.Row():
            M17 = gr.Textbox(label="主角描述17，例如：lala-穿着红色旗袍的女孩", lines=1, elem_id=self.elem_id("M17"))
            M18 = gr.Textbox(label="主角描述18，例如：jack-穿着绿色毛衣的男孩", lines=1, elem_id=self.elem_id("M18"))
            M19 = gr.Textbox(label="主角描述19，例如：maka-身上破烂不堪的僵尸", lines=1, elem_id=self.elem_id("M19"))
            M20 = gr.Textbox(label="主角描述20，例如：sara-穿花衬衫的30岁大叔", lines=1, elem_id=self.elem_id("M20"))

        with gr.Row():
            CX2 = gr.Checkbox(label="黑白漫画模式【开启后会输出为漫画的图】", value=False, display="inline-block", elem_id=self.elem_id("CX2"))
            CX3 = gr.Checkbox(label="彩色漫画模式【开启后，会输出彩漫，会覆盖黑白】", value=True, display="inline-block", elem_id=self.elem_id("CX3"))
            CX5 = gr.Checkbox(label="插画模式【开启后，会输出彩色插画，会覆盖彩漫】", value=False, display="inline-block", elem_id=self.elem_id("CX5"))
        ttm = gr.Slider(minimum=-6, maximum=6, step=1, label='写实程度【往左边是卡通，往右边是写实】', value=0, elem_id=self.elem_id("ttm"))
        with gr.Row():
            txtt = gr.Slider(minimum=0, maximum=6, step=1, label='透视强度【越强透视越狠】', value=0, elem_id=self.elem_id("txtt"))
            fast = gr.Slider(minimum=0, maximum=6, step=1, label='动态强度【值越大，画面越动感，太大角色会崩】', value=0, elem_id=self.elem_id("fast"))
        with gr.Row():
            ow_text = gr.Textbox(label="其他画风（输入画风，构图等等控制）最上面不选择的时候输入", lines=1, elem_id=self.elem_id("CX4"))
            style_txt = gr.Textbox(label="其他时代背景（时间，时代背景等.）最上面不选择的时候输入", lines=1, elem_id=self.elem_id("style"))
        flow_text = gr.Textbox(label="其他（可以补充任意词，会对每张图产生作用，比如lora之类的）", lines=1, elem_id=self.elem_id("flow_text"))
        file = gr.File(label="上传文件来载入任务列表，注意每行会产生一张图的任务。", type='binary', elem_id=self.elem_id("file"))
        file.change(fn=load_prompt_file, inputs=[file], outputs=[file, PTX, PTX])
        PTX.change(lambda tb: gr.update(lines=7) if ("\n" in tb) else gr.update(lines=2), inputs=[PTX], outputs=[PTX])
        return [PTX, style_txt, flow_text, ow_text, seedX, CX1, CX2, CX3, CXt, MXT, fast, ttm, tttr, txtt, xtf, rtf, CX5, CXx, M1, M2, M3, M4, M5, M6,
                M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, M17, M18, M19, M20]

    def run(self, p, PTX: str, style_txt: str, flow_text: str, ow_text: str, seedX: int, CX1: bool, CX2: bool, CX3: bool, CXt: bool, MXT: int,
            fast: int, ttm: int, tttr: int, txtt: int, xtf: str, rtf: str, CX5: bool, CXx: bool, M1: str, M2: str, M3: str, M4: str, M5, M6,
            M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, M17, M18, M19, M20):
        global ppt, dddf

        qdf = f",{PTX},"
        if CXx:
            qdf = re.split(r'[.]', qdf)
        else:
            qdf = qdf.split(',')
        lines = [x.strip() for x in qdf if x.strip()]

        if not CXx:
            lines = [x.strip() for x in PTX.split('\n') if x.strip()]

        lines = [x for x in lines if len(x) > 0]
        p.do_not_save_grid = True
        ppt = f",{style_txt},"
        dddf = f",{flow_text},"
        mtt = f",{ow_text},"
        c1 = f"{M1},"
        c2 = f"{M2},"
        c3 = f"{M3},"
        c4 = f"{M4},"
        c5 = f"{M5},"
        c6 = f"{M6},"
        c7 = f"{M7},"
        c8 = f"{M8},"
        c9 = f"{M9},"
        c10 = f"{M10},"
        c11 = f"{M11},"
        c12 = f"{M12},"
        c13 = f"{M13},"
        c14 = f"{M14},"
        c15 = f"{M15},"
        c16 = f"{M16},"
        c17 = f"{M17},"
        c18 = f"{M18},"
        c19 = f"{M19},"
        c20 = f"{M20},"
        jobs = []
        job_count = 0
        # mcc = get_name(stf)
        ddc = '(' + get_name(rtf) + ')'

        global ooo
        if "hair" in lines or "cloth" in lines:
            ooo = ""

        ggg = ooo + ppt + dddf

        if CX2:
            ggg = ggg + mtt1
            mcc = ""
        if CX3:
            ggg = ggg + mtt2
        if CX5:
            ggg = ggg + mtt3

        lines = lines * MXT

        for line in lines:
            args = {"prompt": line + '((' + get_name(xtf) + '))' + mtt + ddc + ggg + tttrt(tttr) + ddd(fast) + qqq(ttm) + ttxq(txtt)}
            job_count += args.get("n_iter", p.n_iter)
            jobs.append(args)

        print(f"准备 处理 {len(lines)} 行 在 {job_count} 任务列表，整个任务开始.")
        if seedX != -1:
            p.seed = seedX

        state.job_count = job_count

        images = []
        all_prompts = []
        infotexts = []
        for n, args in enumerate(jobs):
            state.job = f"{state.job_no + 1} out of {state.job_count}"
            if CX1:
                args["prompt"] = "((" + random.choice(cccctt.split(",")) + "))" + "," + args["prompt"]
            if CXt:
                args["prompt"] = "((" + random.choice(mqfd.split(",")) + "))" + "," + args["prompt"]

            args["prompt"] = kkkk(args["prompt"],
                                  c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10 + c11 + c12 + c13 + c14 + c15 + c16 + c17 + c18 + c19 + c20)
            print("-----准备开始处理任务：", n + 1)
            copy_p = copy.copy(p)
            for k, v in args.items():
                setattr(copy_p, k, v)

            proc = process_images(copy_p)
            images += proc.images
            print("   处理了一张图. seed值为:", copy_p.seed)

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
