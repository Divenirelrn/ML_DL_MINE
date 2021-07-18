from __future__ import division, print_function, absolute_import
import pickle
import numpy as np 
import selectivesearch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os.path
import skimage
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
import preprocessing_RCNN as prep
import os

from plot import image_plot

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Load testing images
def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img

def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")

def image_proposal(img_path):
    img = skimage.io.imread(img_path)
    img_lbl, regions = selectivesearch.selective_search(
                       img, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    images = []
    vertices = []
    for r in regions:
	# excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        if r['size'] < 220:
            continue
	# resize to 224 * 224 for input
        proposal_img, proposal_vertice = prep.clip_pic(img, r['rect'])
        # Delete Empty array
        if len(proposal_img) == 0:
            continue
        # Ignore things contain 0 or not C contiguous array
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        # Check if any 0-dimension exist
        [a, b, c] = np.shape(proposal_img)
        if a == 0 or b == 0 or c == 0:
            continue
        im = Image.fromarray(proposal_img)
        resized_proposal_img = resize_image(im, 224, 224)
        candidates.add(r['rect'])
        img_float = pil_to_nparray(resized_proposal_img)
        images.append(img_float)
        vertices.append(r['rect'])
    return images, vertices

# Load training images
def generate_single_svm_train(one_class_train_file):
    trainfile = one_class_train_file
    savepath = one_class_train_file.replace('txt', 'pkl')
    images = []
    Y = []
    if os.path.isfile(savepath):
        print("restoring svm dataset " + savepath)
        images, Y = prep.load_from_pkl(savepath)
    else:
        print("loading svm dataset " + savepath)
        images, Y = prep.load_train_proposals(trainfile, 2, threshold=0.3, svm=True, save=True, save_path=savepath)
    return images, Y
    
# Use a already trained alexnet with the last layer redesigned
def create_alexnet(num_classes, restore=False):
    # Building 'AlexNet'
    network = input_data(shape=[None, 224, 224, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network

# Construct cascade svms

def train_svms(train_file_folder, model):
    listings = os.listdir(train_file_folder)
    svms = []
    for train_file in listings:
        if "pkl" in train_file:
            continue
        X, Y = generate_single_svm_train(train_file_folder+train_file)
        train_features = []
        for i in X:
            feats = model.predict([i])
            train_features.append(feats[0])
        print("feature dimension")
        print(np.shape(train_features))
        svm_i = svm.LinearSVC()
        clf = CalibratedClassifierCV(svm_i) 
        print("fit svm")
        clf.fit(train_features, Y)
        #y_proba = clf.predict_proba(X_test)
        #clf = svm.LinearSVC()
        #clf.fit(train_features, Y)
        svms.append(clf)
    return svms

if __name__ == '__main__':
    import pickle
    import cv2

    train_file_folder = 'svm_train/'
    img_dir = '2flowers/jpg/0'
    net = create_alexnet(3)
    model = tflearn.DNN(net)
    model.load('models/fine_tune_model_save.model')
    svm_ckpt = "models/svms.pickle"
    if os.path.exists(svm_ckpt):
        with open(svm_ckpt, "rb") as fp:
            svms = pickle.load(fp)
    else:
        svms = train_svms(train_file_folder, model)
        with open(svm_ckpt, "wb") as fp:
            pickle.dump(svms, fp)
    print("Done fitting svms")
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        imgs, verts = image_proposal(img_path)
        features = model.predict(imgs)
        print("predict image:")
        print(np.shape(features))
        results = []
        results_label = []
        count = 0
        for f in features:
            #print("f:", f.shape)
            #print("f:", type(f))
            f = f[np.newaxis, ...]
            for i in svms:
                pred = i.predict(f)
                if pred[0] != 0:
                    scores = i.predict_proba(f)
                    score = round(float(scores.max()), 2)
                    cls = int(scores.argmax())
                    results.append([verts[count], cls, score])
                    results_label.append(pred[0])
            count += 1
        print("result:")
        print(results)
        print("result label:")
        print(results_label)
        show = False
        save = True
        if save:
            if not os.path.exists("./results"):
                os.makedirs("./results")

            if len(results):
                results = sorted(results, key=lambda x: x[2], reverse=True)
                results = [results[0]]
            img = cv2.imread(img_path)
            image_plot(img, results, img_name)

        if show:
            img = skimage.io.imread(img_path)
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
            ax.imshow(img)
            for x, y, w, h in results:
                rect = mpatches.Rectangle(
                    (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
                ax.add_patch(rect)

            plt.show()










