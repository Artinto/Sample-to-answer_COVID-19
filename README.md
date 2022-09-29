# Sample-to-answer Platform for the Clinical Evaluation of COVID-19 using a Deep-learning-assisted Smartphone-based Assay
## I. Enviroment setting
#### ✔ Python 3.8 and Ubuntu 16.04 are required
### 1. git install
```bash
$ sudo apt-get install git

$ git config --global user.name <user_name>
$ git config --global user.email <user_email>
```

<br>

### 2. Clone this Repository on your local path
```bash
$ cd <your_path>
$ git clone https://github.com/Artinto/Diagnosis_Kit
```

<br>

### 3. Create the virtual enviroment (optional)
        
#### - install virtualenv
```bash
$ sudo apt-get install virtualenv
```

#### - Create your virtual enviroment (insert your venv_name)
```bash
$ virtualenv <venv_name>  --python=python3.8
```

#### - Activate your virtual environment
```bash
$ . <venv_name>/bin/activate
```
**&rarr; Terminal will be...**   ```(venv_name) $ ```
  
#### -  Install requirements packages
```bash
$ pip install -r requirements.txt
```

<br>

## II. Download Weight file (Please download right directory)
- yolo/weights/[yolov3_tline.weights](https://drive.google.com/file/d/1QTrlcYSU8M6GGecWqhspaNu0iBD9__RJ/view?usp=sharing)

- classification/weights/[bce_weight.pth](https://drive.google.com/file/d/1L7DCQpbuqNUR-hDSy4NdsYtFJbCJj1m-/view?usp=sharing)

- regression/weights/[reg_weight.pth](https://drive.google.com/file/d/1uXvbIWH51fu283BL7TUqmzuJOtjSvZ8l/view?usp=sharing)

<br>

## III. Crop the smartphone image using YOLOv3
In this process, You can see the cropped image. Actually, cropped data is directly passed to classifier(ResNet18). But this code consist 2-stages to show the intermediate processes. **Images are 100 each class in 'yolo/images/' folder.** 
We used more data in the paper. If you need, download this [link](https://drive.google.com/file/d/1wq5-V3CD3OE3TdBWT1oZ15-qrFHPJqlD/view?usp=sharing). 
	
```bash
$ cd yolo

$ python crop.py        (this command takes a few minutes)
```
**&rarr; Check the 'yolo/result/' folder**

<br>

## IV. Classify negative/positive cropped images using ResNet18
You can see the classification result for the each 'yolo/result/' image.

```bash
$ cd ../classification/

$ python test.py
```	
**&rarr; Check the 'classification/result/result.txt'**

<br>

## V. Concentration Regression using ResNet50
You need to check the trend with increasing concentration. The higher the density, the higher the number. The x-axis represents the correct answer and the y-axis represents the predicted value.

```bash
$ cd ../regression/

$ python test.py
```
**&rarr; Check the 'regression/result/result.png'**

<br>

## VI. Pipeline mode (Additional)
In this mode, You can only view the results without saving any images. Cropped images from YOLO are send directly to ResNet. Total Accuracy is printed on Terminal, detailed result is saved on 'classfication/result/matrix.txt'.

```bash
$ cd ../yolo/

$ python crop.py pipeline        (this command takes a few minutes)
```
**&rarr; Check the 'classification/result/matrix.txt' file**
