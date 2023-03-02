This is a guide of this code. Please follow the next steps.


I. Enviroment setting
   *Python 3.8 and Ubuntu 16.04 are required*   

   1) Unzip the zip file. (Type the text below on Terminal.)

	$ unzip covid_with_DL.zip

   2) Change directory to the unzipped folder

   3) Create the virtual enviroment (optional)
        
      - Install virtualenv

	$ sudo apt-get install virtualenv


      - Create your virtual enviroment (insert your venv_name)
	
	$ virtualenv <venv_name>  --python=python3.8


      - Activate your virtual environment

	$ . <venv_name>/bin/activate

	--> Terminal will be...   (venv_name) $ 


   4) Install requirements packages
	
	(venv_name) $ pip install -r requirements.txt



II. Downloading Weight files to the following destinations
      
      - yolo/weights/yolov3_tline.weights

(https://drive.google.com/file/d/1QTrlcYSU8M6GGecWqhspaNu0iBD9__RJ/view?usp=sharing)


      - classification/weights/bce_weight.pth

(https://drive.google.com/file/d/1L7DCQpbuqNUR-hDSy4NdsYtFJbCJj1m-/view?usp=sharing)



III. Cropping the smartphone image through YOLOv3
      100 test sample images for each class are provided in 'yolo/images/' folder. If you need more test samples, please download through this link. (https://drive.google.com/file/d/1wq5-V3CD3OE3TdBWT1oZ15-qrFHPJqlD/view?usp=sharing)

	

	(venv_name) $ cd yolo

	(venv_name) $ python crop.py        (this command takes a few minutes)

	--> Check the 'yolo/result/' folder



IV. Classification (negative/positive) of the cropped images through ResNet18
	You can see the classification result.
	(input: the cropped images in 'yolo/result/')
	
	(venv_name) $ cd ../classification/

	(venv_name) $ python test.py
	
	--> Check the 'classification/result/result.txt'



V. Pipeline mode (Additional)
	In this mode, you can check the final result without saving any images. The cropped images from Yolov3 are directly sent to ResNet. The brief summary about accuracy is printed on the console screen and the corresponding confusion matrix is written in 'classfication/result/matrix.txt'.

	(venv_name) $ cd ../yolo/

	(venv_name) $ python crop.py pipeline        (this command takes a few minutes)

	--> Check the 'classification/result/matrix.txt' file

