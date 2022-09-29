import torch
import matplotlib.pyplot as plt
from PIL import Image
from model import REG
from utils import *

# load weight and convert to evaluate mode
net = REG()
net.init_weights()
net.load_state_dict(torch.load('./weights/reg_weight.pth'))
net.eval()

# src: cropped images, dst: save result of test
src = './images/'
dst = './result/'

# load image data
test_loader = data_load(src)

# test the dilution images
x = []
y = []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data, target
        output = net(data)
        target = target.unsqueeze(1)
        target = target.float()

        for tar, out in zip(target, output):
            x.append(float(round(tar.item(), 1)))
            y.append(float(round(out.item(), 3)))

# save the result graph
auto_folder(dst)
plt.plot(x, y, 'o')
plt.xlim([0, 12])
plt.ylim([0, 12])
plt.savefig(dst + 'result.png')
