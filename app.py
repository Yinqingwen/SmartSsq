# -*- coding:utf-8 -*-
"""
Author: yinqingwen
Email:yinqingwen@gmail.com
"""

import json
import time
import datetime
import numpy as np
import tensorflow as tf
from config import *
from flask import Flask
from get_train_data import get_current_number, spider, pd

app = Flask(__name__)

# 关闭eager模式
tf.compat.v1.disable_eager_execution()

red_graph = tf.compat.v1.Graph()
with red_graph.as_default():
    red_saver = tf.compat.v1.train.import_meta_graph("{}red_ball_model.ckpt.meta".format(red_ball_model_path))
red_sess = tf.compat.v1.Session(graph=red_graph)
red_saver.restore(red_sess, "{}red_ball_model.ckpt".format(red_ball_model_path))
print("[INFO] 已加载红球模型！")

blue_graph = tf.compat.v1.Graph()
with blue_graph.as_default():
    blue_saver = tf.compat.v1.train.import_meta_graph("{}blue_ball_model.ckpt.meta".format(blue_ball_model_path))
blue_sess = tf.compat.v1.Session(graph=blue_graph)
blue_saver.restore(blue_sess, "{}blue_ball_model.ckpt".format(blue_ball_model_path))
print("[INFO] 已加载蓝球模型！")

# 加载关键节点名
with open("{}{}".format(model_path, pred_key_name)) as f:
    pred_key_d = json.load(f)

app = Flask(__name__)

@app.route('/')
def main():
    #往期期数
    diff_number = windows_size - 1
    data = spider(str(int(get_current_number()) - diff_number), get_current_number(), "predict")
    #红球名称列表
    red_name_list = [(BOLL_NAME[0], i + 1) for i in range(sequence_len)]
    #红球数据列表
    red_data = data[["{}号码_{}".format(name[0], i) for name, i in red_name_list]].values.astype(int) - 1
    #篮球数据列表
    blue_data = data[[BOLL_NAME[1][0]]].values.astype(int) #- 1
    print(blue_data)
    return "diff_number.to_bytes"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)