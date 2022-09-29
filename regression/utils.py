import os
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def auto_folder(directory):  # createFolder
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


def get_image_list(path):
    file_dir = path
    file_list = os.listdir(path)
    img_list = []

    for file in file_list:
        if file.endswith(".jpg") or file.endswith(".png"):
            img_list.append(os.path.join(file_dir, file))
        else:
            img_list.extend(get_image_list(os.path.join(file_dir, file)))

    return img_list


def data_load(dataroot_test):
    batch_size = 8

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((270, 375)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = dset.ImageFolder(root=dataroot_test, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return testloader
