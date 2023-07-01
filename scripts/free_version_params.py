env_data = {
    # 百度翻译相关开始 在此查看如何获取 https://fanyi-api.baidu.com/doc/11
    'BAIDU_TRANSLATE_APPID': '',
    'BAIDU_TRANSLATE_KEY': '',
    # 百度翻译相关结束
    # gpt登陆后的ACCESS_TOKEN 在此查看你登陆后的access token  https://chat.openai.com/api/auth/session
    'ACCESS_TOKEN': '',
    # gpt的api的key 在此查找你的gpt的apikey https://platform.openai.com/account/api-keys
    'KEY': '',
    # gpt 反向代理的api请求地址，不设置此项需要在.env文件中配置代理，并启用代理，需要电脑本身有代理，需要有自己的服务器以及域名设置难度较大
    'API_URL': '',
    # gpt 反向代理的web请求方式的代理地址，不设置此项需要在.env文件中配置代理，并启用代理，需要电脑本身有代理 需要有自己的服务器以及域名设置难度较大
    'CHATGPT_BASE_URL': '',
    # 首先要使用语音功能，请配置至少一项语音的内容，否则无法使用语音功能
    # 百度语音相关开始 可为空 为空时无法使用百度语音 在此了解如何获取 https://ai.baidu.com/ai-doc/SPEECH/qknh9i8ed
    'BAIDU_VOICE_API_KEY': '',
    'BAIDU_VOICE_SECRET_KEY': '',
    'BAIDU_VOICE_APP_ID': '',
    # 百度语音相关开始
    # 阿里云相关开始 可为空 为空时无法使用阿里语音 在此了解如何获取 https://help.aliyun.com/document_detail/72138.html?spm=5176.12061031.J_5253785160.6.22306822CvKjC6
    'ALIYUN_ACCESSKEY_ID': '',
    'ALIYUN_ACCESSKEY_SECRET': '',
    'ALIYUN_APPKEY': '',
    # 阿里云相关结束
    # 华为相关开始 可为空 为空时无法使用华为语音 在此了解如何获取 https://support.huaweicloud.com/productdesc-sis/sis_01_0001.html
    'HUAWEI_AK': '',
    'HUAWEI_SK': '',
    # 华为相关结束
    # azure的speech key 可为空，为空时无法使用微软语音 在此了解如何获取 https://portal.azure.com/#home 需要有微软账号以及开通azure中的语音服务
    'AZURE_SPEECH_KEY': '',
    # deepl翻译的apikey 可为空，为空时将使用百度翻译来翻译AI推文，效果没有这个好 在此了解如何获取 https://www.deepl.com/pro-api?cta=header-pro-api
    'DEEPL_API_KEY': '',
    # 是否是 gpt的plus用户，默认False
    'IS_AI_PLUS': False,
}


def validate_parameters(data):
    allowed_empty_keys = {
        'API_URL',
        'CHATGPT_BASE_URL',
        'DEEPL_API_KEY',
        'ALIYUN_ACCESSKEY_ID',
        'ALIYUN_ACCESSKEY_SECRET',
        'ALIYUN_APPKEY',
        'BAIDU_VOICE_API_KEY',
        'BAIDU_VOICE_SECRET_KEY',
        'BAIDU_VOICE_APP_ID',
        'HUAWEI_AK',
        'HUAWEI_SK',
        'AZURE_SPEECH_KEY',
        'DEEPL_API_KEY'
    }
    empty_keys = []

    for key, value in data.items():
        if value == '' and key not in allowed_empty_keys:
            print(f"{key} 是空的")
            empty_keys.append(key)

    if empty_keys:
        print("所有为空的参数：", ", ".join(empty_keys))
        return False
    else:
        print("参数有效")
        return True
