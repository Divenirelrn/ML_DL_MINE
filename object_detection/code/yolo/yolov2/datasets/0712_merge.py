import os
import shutil	
from tqdm import tqdm
	
voc_07_dir = '/home/data/train_places/data/VOCdevkit/VOC2007'
voc_12_dir = '/home/data/train_places/data/VOCdevkit/VOC2012'

anno_07_dir = os.path.join(voc_07_dir, 'Annotations')
text_07_dir = os.path.join(voc_07_dir, 'ImageSets/Main')
img_07_dir = os.path.join(voc_07_dir, 'JPEGImages')

anno_12_dir = os.path.join(voc_12_dir, 'Annotations')
text_12_dir = os.path.join(voc_12_dir, 'ImageSets/Main')
img_12_dir = os.path.join(voc_12_dir, 'JPEGImages')

anno_list = os.listdir(anno_12_dir)
for anno in tqdm(anno_list):
    anno_path = os.path.join(anno_12_dir, anno)
    dst_path = os.path.join(anno_07_dir, anno)
    shutil.copy(anno_path, dst_path)

print('Annotations copy over!')

text_list = os.listdir(text_12_dir)
for text in tqdm(text_list):
    text_path = os.path.join(text_12_dir, text)
    with open(text_path, 'r') as fp:
        data1 = fp.readlines()
    #print('len data1:', len(data1))

    dst_path = os.path.join(text_07_dir, text)
    with open(dst_path, 'r') as fp:
        data2 = fp.readlines()
    #print('len data2:', len(data2))

    data = data1 + data2
    #print('len data:', len(data))
    with open(dst_path, 'w') as fo:
        for line in data:
            fo.write(line)

print('Text copy over!')

img_list = os.listdir(img_12_dir)
for img in tqdm(img_list):
    img_path = os.path.join(img_12_dir, img)
    dst_path = os.path.join(img_07_dir, img)
    shutil.copy(img_path, dst_path)

print('Image copy over!')
