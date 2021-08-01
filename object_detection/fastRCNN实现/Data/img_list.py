import os
import random

img_list = os.listdir("Images")

"""
with open("all_list.txt", "w") as fp:
    for img in img_list:
        name = img.split(".")[0]
        fp.write(name + "\n")
"""

n = len(img_list)
train_ratio = 0.7
train_count = int(n * train_ratio)
random.shuffle(img_list)

with open("train_list.txt", "w") as fp:
    for img in img_list[:train_count]:
        name = img.split(".")[0]
        fp.write(name + "\n")

with open("test_list.txt", "w") as fp:
    for img in img_list[train_count:]:
        name = img.split(".")[0]
        fp.write(name + "\n")
