# coding=utf-8

import http.client
import hashlib
import urllib
import random
import json

# https://fanyi-api.baidu.com/api/trans/product/apidoc
#百度通用翻译API,不包含词典、tts语音合成等资源，如有相关需求请联系translate_api@baidu.com

# appid = '20231107001872574'  # 填写你的appid
# secretKey = 'JY_4DXaqILwmbq424Ise'  # 填写你的密钥

# httpClient = None
# myurl = '/api/trans/vip/translate'

# fromLang = 'auto'   #原文语种
# toLang = 'zh'   #译文语种
# salt = random.randint(32768, 65536)
# q= "XYZ started working as a courier for the company and soon became a part of the company's infrastructure, as well as a member of the company's staff."
# sign = appid + q + str(salt) + secretKey
# sign = hashlib.md5(sign.encode()).hexdigest()
# myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
# salt) + '&sign=' + sign

# try:
#     httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
#     httpClient.request('GET', myurl)

#     # response是HTTPResponse对象
#     response = httpClient.getresponse()
#     result_all = response.read().decode("utf-8")
#     result = json.loads(result_all)

#     print (result)
#     print(result['trans_result'][0]['dst'])

# except Exception as e:
#     print (e)
# finally:
#     if httpClient:
#         httpClient.close()

# Baidu API
def translate(q, fromLang = 'en', toLang = 'zh'):
    appid = '20231107001872574'  # 填写你的appid
    secretKey = 'JY_4DXaqILwmbq424Ise'  # 填写你的密钥
    myurl = '/api/trans/vip/translate'
    salt = random.randint(32768, 65536)
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
    salt) + '&sign=' + sign
    print('src:', q)

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)

        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)

        # print (result)
        print('dst:', result['trans_result'][0]['dst'])

        if httpClient:
            httpClient.close()

        return result['trans_result'][0]['dst']

    except Exception as e:
        print ('err:', e)
        print ('rsp:', result)
        if httpClient:
            httpClient.close()
        return -1

import itertools
import time
# translate with BAIDU API
def translate_annotations():
    # metrics = ['regard', 'sentiment']
    metrics = ['sentiment']
    files = ['train_other', 'dev', 'test', 'train']
    for metric, file in itertools.product(metrics, files):
        print(metric, file)
        translate_data = []
        with open(f'./data/{metric}/{file}.tsv', 'r') as f, open(f'./data/{metric}-chinese/{file}.tsv', 'w') as f2:
            for line in f:
                score, sentence = line.strip().split('\t')
                translate_sentence = translate(sentence)
                f2.write(score + '\t' + translate_sentence + '\n')
                time.sleep(1)
        print(f'translate data {metric}/{file}: {len(translate_data)}')


# pip install goslate
# import goslate
# gs = goslate.Goslate()
# print(gs.translate('hello world', 'zh'))

# using google translate
import goslate
def translateTextList(text_list):
    gs = goslate.Goslate()
    gs.translate('hello world', 'zh')

import goslate
# translate with Google API
def translate_annotations2():
    gs = goslate.Goslate()
    metrics = ['sentiment']
    files = ['train_other', 'dev', 'test', 'train']
    for metric, file in itertools.product(metrics, files):
        print(metric, file)
        translate_data = []
        with open(f'./data/{metric}/{file}.tsv', 'r') as f, open(f'./data/{metric}-chinese/{file}.tsv', 'w') as f2:
            for line in f:
                score, sentence = line.strip().split('\t')
                translate_sentence = gs.translate(sentence, 'zh')
                f2.write(score + '\t' + translate_sentence + '\n')
                time.sleep(0.1)
        print(f'translate data {metric}/{file}: {len(translate_data)}')

import goslate
# translate Samples
def translateSamples():
    gs = goslate.Goslate()
    files = [
        'sample.tsv',
        'small_gpt2_generated_samples.tsv',
    ]
    for file in files:
        print(file)
        translate_data = []
        with open(f'./data/generated_samples/{file}', 'r') as f, open(f'./data/generated_samples_zh/{file}', 'w') as f2:
            for line in f:
                sentence = line.strip()
                translate_sentence = gs.translate(sentence, 'zh')
                f2.write(translate_sentence + '\n')
                time.sleep(0.1)
        print(f'translate data {file}: {len(translate_data)}')


if __name__ == '__main__':
    # translate("XYZ started working as a courier for the company and soon became a part of the company's infrastructure, as well as a member of the company's staff.")
    # translate_annotations()
    # translate_annotations2()
    translateSamples()
    pass
