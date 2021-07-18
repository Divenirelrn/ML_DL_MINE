import cv2
import random

def image_plot(img, results, img_name):
    colors = [[100, 0, 0], [0,0,100], [0, 100, 0], [0,0,205], [224,255,255],
              [238,173,14],[255,255,0],[255,106,106],[255,0,0],[255,64,64]]
    for res in results:
        color = random.choice(colors)
        loc, cls, score = res
        c1 = (loc[0], loc[1])
        c2 = (loc[2], loc[3])
        label = str(cls) + ":" + str(score)

        cv2.rectangle(img, c1, c2, color)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)

    cv2.imwrite("./results/" + img_name, img)
