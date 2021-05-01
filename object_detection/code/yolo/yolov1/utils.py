#-*-coding:utf-8-*-
import os
import multiprocessing as mp
#import sys


#sys.stdout.encoding = utf-8


with open("/tmp/listfile.txt", "r") as f:
    lines = f.readlines()
vocall = [line.rstrip("\n") for line in lines]


#with open("./voc2012.txt", "r") as f:
#    lines = f.readlines()
#voc12 = [line.rstrip("\n") for line in lines]


#vocall = voc07 + voc12
vocall = [line.split(" ")[0] for line in vocall]


img_list = os.listdir("../../data/vocall")


def worker(img_name, q):
    if img_name not in img_list:
        print(os.getpid(), img_name)
    q.put(img_name)


def main():
    po = mp.Pool(10)
    q = mp.Manager().Queue()
    for img in vocall:
        po.apply_async(worker, (img,q))

    po.close()
    # po.join()
    length = len(img_list)
    num = 0
    for i in range(len(img_list)):
        q.get()
        num += 1
        print("\r当前进度：%.2f %%" % (100 * num / length), end="")
    print("\n")



if __name__ == "__main__":
    main()
