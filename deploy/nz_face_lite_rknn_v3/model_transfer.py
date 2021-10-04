import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

ONNX_MODEL = 'fatigue_mobileV3_small_torch16.onnx'
RKNN_MODEL = 'fatigue_recog.rknn'
#ONNX_MODEL = 'resnet50.onnx'
#RKNN_MODEL = 'resnet50.rknn'
#ONNX_MODEL = 'alexnet.onnx'
#RKNN_MODEL = 'alexnet.rknn'


def show_outputs(outputs):
    output = outputs[0][0]
    output_sorted = sorted(output, reverse=True)
    top5_str = 'mobilenetV3\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


def readable_speed(speed):
    speed_bytes = float(speed)
    speed_kbytes = speed_bytes / 1024
    if speed_kbytes > 1024:
        speed_mbytes = speed_kbytes / 1024
        if speed_mbytes > 1024:
            speed_gbytes = speed_mbytes / 1024
            return "{:.2f} GB/s".format(speed_gbytes)
        else:
            return "{:.2f} MB/s".format(speed_mbytes)
    else:
        return "{:.2f} KB/s".format(speed_kbytes)


def show_progress(blocknum, blocksize, totalsize):
    speed = (blocknum * blocksize) / (time.time() - start_time)
    speed_str = " Speed: {}".format(readable_speed(speed))
    recv_size = blocknum * blocksize

    f = sys.stdout
    progress = (recv_size / totalsize)
    progress_str = "{:.2f}%".format(progress * 100)
    n = round(progress * 50)
    s = ('#' * n).ljust(50, '-')
    f.write(progress_str.ljust(8, ' ') + '[' + s + ']' + speed_str)
    f.flush()
    f.write('\r\n')


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[[107.327034, 103.71193, 98.97443]], std_values=[[72.68962, 72.23954, 72.20405]], reorder_channel='0 1 2', target_platform=['rv1109'])
    print('done')

    # Load tensorflow model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, pre_compile=True, dataset='./dataset.txt')
    #ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    #ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export mobilenetV3.rknn failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('./images/f1.jpg')
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    x = outputs[0]
    print("x:", x)
    output = np.exp(x)/np.sum(np.exp(x))
    #outputs = [output[0]]
    print("output:", output[0])
    #print("output:", output[0].argmax(dim=1))
    #show_outputs(outputs)
    print('done')

    rknn.release()