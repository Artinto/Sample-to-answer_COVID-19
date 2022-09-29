from utils import *
from yolo import process
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from classification.model import BCE


# enter original image directory(src) and crop image directory(dst)
src = './images/'
dst = './result/'

# get original image list
img_list = get_image_list(src)

# pipeline mode: load BCE model
net = BCE()
net.load_state_dict(torch.load('../classification/weights/bce_weight.pth'))
net.eval()

# enter original image to yolo network and save the cropped image
matrix_confusion = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
cls_dict = {0: 'negative', 1: 'positive'}

for idx, img in enumerate(img_list):
    with open(img, 'rb') as f:
        data = f.read()

    # RGB image
    rotate_image, valid = process(data)

    # any box detect
    if not valid:
        continue

    # debug mode (default)
    if len(sys.argv) == 1:
        s = img.split('/')
        ex_path = '/'.join(s[len(src.split('/')) - 1:-1])
        file_name = s[-1][:-4] + '_crop' + s[-1][-4:]
        save_path = os.path.join(dst, ex_path)
        auto_folder(save_path)

        bgr_image = cv2.cvtColor(rotate_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, file_name), bgr_image)

    # pipeline mode
    elif sys.argv[1] == 'pipeline':
        # convert cv2 to pil
        pil_image = Image.fromarray(rotate_image)

        # convert pil to tensor
        tensor = transform_image(pil_image)

        with torch.no_grad():
            output = net.forward(tensor)

        prediction = int(torch.round(output))

        s = img.split('/')
        cls = s[len(src.split('/')) - 1]

        if cls_dict[prediction] == cls:
            result = 'True'

            if cls == 'positive':
                matrix_confusion['TP'] += 1
            elif cls == 'negative':
                matrix_confusion['TN'] += 1
        else:
            result = 'False'

            if cls == 'positive':
                matrix_confusion['FP'] += 1
            elif cls == 'negative':
                matrix_confusion['FN'] += 1

    else:
        print("Invalid argument!")
        break

    msg = '\rProgress %d%%' % ((idx + 1) / len(img_list) * 100)
    print(msg, end='')


if sys.argv[1] == 'pipeline':
    auto_folder('../classification/result/')
    f = open('../classification/result/matrix.txt', 'w')
    f.write(''.center(10) + 'Actual'.center(20) + '\n')
    f.write('predict'.ljust(10) + '|' + 'positive'.center(10) + 'negative'.center(10) + '\n')
    f.write('-------------------------------\n')
    f.write('positive'.ljust(10) + '|' + str(matrix_confusion['TP']).center(10) + str(matrix_confusion['FP']).center(10) + '\n')
    f.write('negative'.ljust(10) + '|' + str(matrix_confusion['FN']).center(10) + str(matrix_confusion['TN']).center(10) + '\n')
    f.close()

    print("\n\nTotal Accuracy: " + str((matrix_confusion['TP']+matrix_confusion['TN'])/sum(matrix_confusion.values())*100) + '%')
    print("\nThe detailed result is saved as 'classification/result/matrix.txt'")
