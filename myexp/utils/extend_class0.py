import os
import glob
import shutil  # 导入shutil模块
import random
import json
import cv2
import os
import numpy as np
# from skimage import exposure
import math
from tqdm import tqdm

# txtlist=glob.glob('/home/underwater/Code/Data/DeepfishDomain/coco/*.txt')
# # txtlist=[os.path.basename(txt) for txt in txtlist]
# for txt in txtlist:
#     with open(txt, "r") as f:  # 打开文件
#         lines = f.readlines()  # 读取每行文件名
#         filenames = [line.strip() for line in lines]  # 去除每行文件名的空格和换行符

#     # with open(os.path.join("/home/underwater/Code/yolov7/DeepfishDomain/pure_txt",txt), "w") as f:  # 创建新文件
#     #     for filename in filenames:
#     #         f.write(os.path.basename(filename) + "\n")  # 写入每个文件名的基本名称到新文件中
#     txtdir = os.path.basename(txt).split('.')[0]
#     txtdir=f'/home/underwater/Code/Data/DeepfishDomain/coco/{txtdir}'
#     if not os.path.exists(txtdir):
#         os.makedirs(txtdir)
#     imagedir=txtdir+'/image_data'
#     if not os.path.exists(imagedir):
#         os.makedirs(imagedir)
#     labeldir=txtdir+'/annotation_data'
#     if not os.path.exists(labeldir):
#         os.makedirs(labeldir)
#     for filename in filenames:  # 遍历每个文件名
#         shutil.copy(os.path.join('/home/underwater/Code/Data/DeepfishDomain/yolo/images',filename), imagedir)  # 将文件复制到新文件夹下
#     for filename in filenames:  # 遍历每个文件名
#         shutil.copy(os.path.join('/home/underwater/Code/Data/DeepfishDomain/yolo/labels',filename.split('.')[0]+'.txt'), labeldir)


def visualization_jsondata():
    # 读取coco格式的test.json
    with open('/home/underwater/Code/Data/DeepfishDomain/coco/test.json', 'r') as f:
        data = json.load(f)

    # 遍历每个图像
    for img in data['images']:
        # 读取图像
        img_path = os.path.join('/home/underwater/Code/Data/DeepfishDomain/images', img['file_name'])
        image = cv2.imread(img_path)

        # 可视化
        for ann in data['annotations']:
            if ann['image_id'] == img['id']:
                bbox = ann['bbox']
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 255, 0), 2)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Open and read the contents of "1.txt"
# Open and read the contents of "1.txt"
# with open("1.txt", "r") as f1:
#     lines1 = f1.readlines()  # Read each line of "1.txt"
#     filenames1 = [line.strip() for line in lines1]  # Remove whitespace and newline characters from each line

# # Open and read the contents of "2.txt"
# with open("2.txt", "r") as f2:
#     lines2 = f2.readlines()  # Read each line of "2.txt"
#     filenames2 = [line.strip() for line in lines2]  # Remove whitespace and newline characters from each line

# # Combine the filenames from both files and shuffle them
# filenames = filenames1 + filenames2
# random.shuffle(filenames)

# # Write the shuffled filenames to "3.txt"
# with open("3.txt", "w") as f3:
#     for filename in filenames:
#         f3.write(filename + "\n")  # Write each filename to a new line in "3.txt"

def tool_0():
    # 分出0，1，2
    import shutil
    test0_List=[file.strip() for file in open("/home/underwater/Code/Data/DeepfishDomain/org/coco_pure/test_0.txt", "r").readlines()]
    test1_List=[file.strip() for file in open("/home/underwater/Code/Data/DeepfishDomain/org/coco_pure/test_1.txt", "r").readlines()]
    test2_List=[file.strip() for file in open("/home/underwater/Code/Data/DeepfishDomain/org/coco_pure/test_2.txt", "r").readlines()]
    train01_List=[file.strip() for file in open("/home/underwater/Code/Data/DeepfishDomain/org/coco_pure/train_01.txt", "r").readlines()]
    train02_List=[file.strip() for file in open("/home/underwater/Code/Data/DeepfishDomain/org/coco_pure/train_02.txt", "r").readlines()]
    train12_List=[file.strip() for file in open("/home/underwater/Code/Data/DeepfishDomain/org/coco_pure/train_12.txt", "r").readlines()]
    train0_List=[]
    train1_List=[]
    train2_List=[]
    for file in train01_List:
        if file not in train02_List:
            train1_List.append(file)
        if file not in train12_List:
            train0_List.append(file)
    for file in train02_List:
        if file not in train01_List:
            train2_List.append(file)
    cls0_List=test0_List+train0_List
    cls0_List=list(set(cls0_List))
    cls1_List=test1_List+train1_List
    cls2_List=test2_List+train2_List
    # for file in cls0_List:
    #     shutil.copy(f'/home/underwater/Code/Data/DeepfishDomain/images/{file}',f'/home/underwater/Code/Data/DeepfishDomain/ColorCast_domain/0/{file}')
    # for file in cls1_List:
    #     shutil.copy(f'/home/underwater/Code/Data/DeepfishDomain/images/{file}',f'/home/underwater/Code/Data/DeepfishDomain/ColorCast_domain/1/{file}')
    # for file in cls2_List:
    #     shutil.copy(f'/home/underwater/Code/Data/DeepfishDomain/images/{file}',f'/home/underwater/Code/Data/DeepfishDomain/ColorCast_domain/2/{file}')
    return cls0_List,cls1_List,cls2_List

def shift_pic_bboxes(img, bboxes):
    '''
    平移后需要包含所有的框
    参考资料：https://blog.csdn.net/sty945/article/details/79387054
    输入：
        img：图像array
        bboxes：该图像包含的所有boundingboxes，一个list，每个元素为[x_min,y_min,x_max,y_max]
                    要确保是数值
    输出：
        shift_img：平移后的图像array
        shift_bboxes：平移后的boundingbox的坐标，list
    '''
    # ------------------ 平移图像 ------------------
    w = img.shape[1]
    h = img.shape[0]

    x_min = w
    x_max = 0
    y_min = h
    y_max = 0
    for bbox in bboxes:
        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(x_max, bbox[3])
        # name = bbox[4]

    # 包含所有目标框的最小框到各个边的距离，即每个方向的最大移动距离
    d_to_left = x_min
    d_to_right = w - x_max
    d_to_top = y_min
    d_to_bottom = h - y_max

    # 在矩阵第一行中表示的是[1,0,x],其中x表示图像将向左或向右移动的距离，如果x是正值，则表示向右移动，如果是负值的话，则表示向左移动。
    # 在矩阵第二行表示的是[0,1,y],其中y表示图像将向上或向下移动的距离，如果y是正值的话，则向下移动，如果是负值的话，则向上移动。
    x = random.uniform(-(d_to_left / 3), d_to_right / 3)
    y = random.uniform(-(d_to_top / 3), d_to_bottom / 3)
    M = np.float32([[1, 0, x], [0, 1, y]])

    # 仿射变换
    shift_img = cv2.warpAffine(img, M,
                               (img.shape[1], img.shape[0]))  # 第一个参数表示我们希望进行变换的图片，第二个参数是我们的平移矩阵，第三个希望展示的结果图片的大小

    # ------------------ 平移boundingbox ------------------
    shift_bboxes = list()
    for bbox in bboxes:
        shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y])

    return shift_img, shift_bboxes

def changeLight( img):
    '''
    adjust_gamma(image, gamma=1, gain=1)函数:
    gamma>1时，输出图像变暗，小于1时，输出图像变亮
    输入：
        img：图像array
    输出：
        img：改变亮度后的图像array
    '''
    flag = random.uniform(0.5, 1.5)  ##flag>1为调暗,小于1为调亮
    return exposure.adjust_gamma(img, flag)


def rotate_img_bboxes( img, bboxes, angle=5, scale=1.):
    '''
    参考：https://blog.csdn.net/saltriver/article/details/79680189
          https://www.ctolib.com/topics-44419.html
    关于仿射变换：https://www.zhihu.com/question/20666664
    输入:
        img:图像array,(h,w,c)
        bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        angle:旋转角度
        scale:默认1
    输出:
        rot_img:旋转后的图像array
        rot_bboxes:旋转后的boundingbox坐标list
    '''
    # ---------------------- 旋转图像 ----------------------
    w = img.shape[1]
    h = img.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)
    # 计算新图像的宽度和高度，分别为最高点和最低点的垂直距离
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # 获取图像绕着某一点的旋转矩阵
    # getRotationMatrix2D(Point2f center, double angle, double scale)
    # Point2f center：表示旋转的中心点
    # double angle：表示旋转的角度
    # double scale：图像缩放因子
    # 参考：https://cloud.tencent.com/developer/article/1425373
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)  # 返回 2x3 矩阵
    # 新中心点与旧中心点之间的位置
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # 仿射变换
    rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
                             flags=cv2.INTER_LANCZOS4)  # ceil向上取整

    # ---------------------- 矫正boundingbox ----------------------
    # rot_mat是最终的旋转矩阵
    # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
    rot_bboxes = list()
    for bbox in bboxes:
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        # name = bbox[4]
        point1 = np.dot(rot_mat, np.array([(x_min + x_max) / 2, y_min, 1]))
        point2 = np.dot(rot_mat, np.array([x_max, (y_min + y_max) / 2, 1]))
        point3 = np.dot(rot_mat, np.array([(x_min + x_max) / 2, y_max, 1]))
        point4 = np.dot(rot_mat, np.array([x_min, (y_min + y_max) / 2, 1]))

        # 合并np.array
        concat = np.vstack((point1, point2, point3, point4))  # 在竖直方向上堆叠
        # 改变array类型
        concat = concat.astype(np.int32)
        # 得到旋转后的坐标
        rx, ry, rw, rh = cv2.boundingRect(concat)
        rx_min = rx
        ry_min = ry
        rx_max = rx + rw
        ry_max = ry + rh
        # 加入list中
        rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

    return rot_img, rot_bboxes

    # 镜像


def flip_pic_bboxes(img, bboxes):
    '''
    参考：https://blog.csdn.net/jningwei/article/details/78753607
    镜像后的图片要包含所有的框
    输入：
        img：图像array
        bboxes：该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
    输出:
        flip_img:镜像后的图像array
        flip_bboxes:镜像后的bounding box的坐标list
    '''
    # ---------------------- 镜像图像 ----------------------
    import copy
    flip_img = copy.deepcopy(img)
    if random.random() < 0.5:
        horizon = True
    else:
        horizon = False
    h, w, _ = img.shape
    if horizon:  # 水平翻转
        flip_img = cv2.flip(flip_img, 1)
    else:
        flip_img = cv2.flip(flip_img, 0)
    # ---------------------- 矫正boundingbox ----------------------
    flip_bboxes = list()
    for bbox in bboxes:
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        # name = bbox[4]
        if horizon:
            flip_bboxes.append([w - x_max, y_min, w - x_min, y_max])
        else:
            flip_bboxes.append([x_min, h - y_max, x_max, h - y_min])

    return flip_img, flip_bboxes

def tool_1():
    visualization=0
    funcnames=['shift','changeLight','rotate','flip']
    funcname_list = random.choices(funcnames, k=607*3)
    # class0数据增广
    cls0List=glob.glob('/home/underwater/Code/Data/DeepfishDomain/ColorCast_domain/0/*.jpg')
    for i,funcname in tqdm(enumerate(funcname_list)):
        path=cls0List[int(i/3)]
        # print(int(i/3))
        img=cv2.imread(path)
        height, width, _ = img.shape
        basename=os.path.basename(path).split('.')[0]
        bboxes=[]
        for bbox in open(f"/home/underwater/Code/Data/DeepfishDomain/labels/{basename}.txt", "r").readlines():
            bbox=bbox.strip().split(' ')[1:]
            x, y, w, h = map(float, bbox)
            x = x * width
            w = w * width
            y = y * height
            h = h * height
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            bboxes.append([x1,y1,x2,y2])
        if funcname=='shift':
            newimg,newbboxes=shift_pic_bboxes(img,bboxes)
        if funcname=='changeLight':
            newimg=changeLight(img)
            newbboxes=bboxes
        if funcname=='rotate':
            newimg, newbboxes1 = rotate_img_bboxes(img, bboxes)
            newheight, newwidth, _ = newimg.shape
            hscale,wscale=height/newheight,width/newwidth
            newimg=cv2.resize(newimg,(width,height))
            newbboxes=[]
            for bbox in newbboxes1:
                x1, y1, x2, y2=bbox
                x1=x1*wscale
                y1=y1*hscale
                x2 = x2 * wscale
                y2 = y2 * hscale
                newbboxes.append([x1,y1,x2,y2])
        if funcname=='flip':
            newimg, newbboxes = flip_pic_bboxes(img, bboxes)
        cv2.imwrite(f'/home/underwater/Code/Data/DeepfishDomain/0_aug/images/{basename}_{funcname}.jpg',newimg)
        f=open(f'/home/underwater/Code/Data/DeepfishDomain/0_aug/labels/{basename}_{funcname}.txt','w')
        for bbox in newbboxes:
            x1, y1, x2, y2=bbox
            w=x2-x1
            h=y2-y1
            x=x1+w/2
            y=y1+h/2
            x=x/width
            y=y/height
            w=w/width
            h=h/height
            f.writelines(f'0 {str(x)} {str(y)} {str(w)} {str(h)}\n')
        if visualization:
            for bbox in bboxes:
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                              (0, 255, 0), 2)
            for bbox in newbboxes:
                cv2.rectangle(newimg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                          (0, 255, 0), 2)
            imgs = np.hstack([img, newimg])

            cv2.namedWindow("mutil_pic", 0)
            cv2.resizeWindow("mutil_pic", 640, 480)
            cv2.imshow("mutil_pic", imgs)
            cv2.waitKey(0)

def write_txt(key,cls0_List,cls1_List,cls2_List,train_List,test_List):
    if isinstance(key,tuple):
        key1,key2=key
        f_train = open(f'/home/underwater/Code/Data/DeepfishDomain/coco/train_{key1}{key2}.txt', 'w')
        f_test = open(f'/home/underwater/Code/Data/DeepfishDomain/coco/test_{key1}{key2}.txt', 'w')
        key_List = locals()[f'cls{key1}_List']+locals()[f'cls{key2}_List']
    else:
        f_train=open(f'/home/underwater/Code/Data/DeepfishDomain/coco/train_{key}.txt','w')
        f_test=open(f'/home/underwater/Code/Data/DeepfishDomain/coco/test_{key}.txt','w')
        key_List=locals()[f'cls{key}_List']
    random.shuffle(key_List)
    for path in train_List:
        if path in key_List:
            f_train.writelines(path+'\n')
    for path in test_List:
        if path in key_List:
            f_test.writelines(path+'\n')

def tool_2():
    cls0_List,cls1_List,cls2_List=tool_0()
    cls0_aug=glob.glob('/home/underwater/Code/Data/DeepfishDomain/0_aug/images/*.jpg')
    cls0_aug_List=[os.path.basename(path) for path in cls0_aug]
    cls0_List=cls0_List+cls0_aug_List
    all_List=cls0_List+cls1_List+cls2_List
    random.shuffle(all_List)
    train_num=int(0.7*len(all_List))
    train_List=all_List[:train_num]
    test_List=all_List[train_num:]
    write_txt(0,cls0_List,cls1_List,cls2_List,train_List,test_List)
    write_txt(1,cls0_List,cls1_List,cls2_List,train_List,test_List)
    write_txt(2,cls0_List,cls1_List,cls2_List,train_List,test_List)
    write_txt((1,2),cls0_List,cls1_List,cls2_List,train_List,test_List)
    write_txt((0,2),cls0_List,cls1_List,cls2_List,train_List,test_List)
    write_txt((0,1),cls0_List,cls1_List,cls2_List,train_List,test_List)

def tool_3():
    target_imgs='/home/underwater/Code/Data/DeepfishDomain/images'
    if os.path.exists(target_imgs):
        shutil.rmtree(target_imgs)
        os.mkdir(target_imgs)
    img_List=glob.glob('/home/underwater/Code/Data/DeepfishDomain/images_pure/*.jpg')\
             +glob.glob('/home/underwater/Code/Data/DeepfishDomain/0_aug/images/*.jpg')
    for img in tqdm(img_List):
        basename=os.path.basename(img)
        shutil.copy(img,f'{target_imgs}/{basename}')
    target_labels = '/home/underwater/Code/Data/DeepfishDomain/labels'
    if os.path.exists(target_labels):
        shutil.rmtree(target_labels)
        os.mkdir(target_labels)
    txt_List = glob.glob('/home/underwater/Code/Data/DeepfishDomain/labels_pure/*.txt') \
               + glob.glob('/home/underwater/Code/Data/DeepfishDomain/0_aug/labels/*.txt')
    for txt in tqdm(txt_List):
        basename = os.path.basename(txt)
        shutil.copy(txt, f'{target_labels}/{basename}')



if __name__=="__main__":
    # 从原来的txt中分出012
    # tool_0()
    # 0类扩增
    # tool_1()
    # 生成训练测试txt文件
    # tool_2()
    # 合并images和labels
    tool_3()
