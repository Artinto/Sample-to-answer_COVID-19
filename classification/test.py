import torch
from PIL import Image
from model import BCE
from utils import *


# load weight and convert to evaluate mode
net = BCE()
net.load_state_dict(torch.load('./weights/bce_weight.pth'))
net.eval()

# src: cropped images, dst: save result of test
src = '../yolo/result/'
dst = './result/'

# get image list and set class dictionary
img_list = get_image_list(src)
cls_dict = {0:'negative', 1:'positive'}

# make result.txt
auto_folder(dst)
f = open(dst + 'result.txt', 'w')

# test and save prediction to result.txt
total_count = 0
correct_count = 0
for img_path in img_list:
    total_count += 1
    img = Image.open(img_path)
    tensor = transform_image(img)

    with torch.no_grad():
        output = net.forward(tensor)

    prediction = int(torch.round(output))

    s = img_path.split('/')
    cls = s[len(src.split('/')) - 1]

    if cls_dict[prediction] == cls:
        result = 'True'
        correct_count += 1
    else:
        result = 'False'

    f.write(f'file name:{s[-1]} / class:{cls} / prediction:{cls_dict[prediction]} / result:{result}\n')

f.write(f'\n\nTotal Accuracy:{correct_count/total_count*100:.1f}\n')
f.close()
