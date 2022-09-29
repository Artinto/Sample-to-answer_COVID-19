This is a guide of this code. If you follow this guide, you can see what results are coming out of code and data.


I. Enviroment setting
   *Python 3.8 and Ubuntu 16.04 are required*

   1) Right-click on the folder you want and open Terminal.
   

   2) Unzip the zip file. (Type the text below on Terminal.)

	$ unzip covid_with_DL.zip


   3) Create the virtual enviroment (optional)
        
      - install virtualenv

	$ sudo apt-get install virtualenv


      - Create your virtual enviroment (insert your venv_name)
	
	$ virtualenv <venv_name>  --python=python3.8


      - Activate your virtual environment

	$ . <venv_name>/bin/activate

	--> Terminal will be...   (venv_name) $ 


   4) Install requirements packages
	
	(venv_name) $ pip install -r requirements.txt



II. Download Weight file (Please download right directory)
      
      - yolo/weights/yolov3_tline.weights

	(https://drive.google.com/file/d/1QTrlcYSU8M6GGecWqhspaNu0iBD9__RJ/view?usp=sharing)


      - classification/weights/bce_weight.pth

	(https://drive.google.com/file/d/1L7DCQpbuqNUR-hDSy4NdsYtFJbCJj1m-/view?usp=sharing)


      - regression/weights/reg_weight.pth

	(https://drive.google.com/file/d/1uXvbIWH51fu283BL7TUqmzuJOtjSvZ8l/view?usp=sharing)



III. Crop the smartphone image using YOLOv3
      In this process, You can see the cropped image. Actually, cropped data is directly passed to classifier(ResNet18). But this code consist 2-stages to show the intermediate processes. Images are 100 each class in 'yolo/images/' folder. We used more data in the paper. If you need, download this link. (https://drive.google.com/file/d/1wq5-V3CD3OE3TdBWT1oZ15-qrFHPJqlD/view?usp=sharing)
	

	(venv_name) $ cd yolo

	(venv_name) $ python crop.py        (this command takes a few minutes)

	--> Check the 'yolo/result/' folder



IV. Classify negative/positive cropped images using ResNet18
	You can see the classification result for the each image in 'yolo/result/'. 
	
	(venv_name) $ cd ../classification/

	(venv_name) $ python test.py
	
	--> Check the 'classification/result/result.txt'



V. Concentration Regression using ResNet50
	You need to check the trend with increasing concentration. The higher the density, the higher the number. The x-axis represents the correct answer and the y-axis represents the predicted value.

	(venv_name) $ cd ../regression/

	(venv_name) $ python test.py
	
	--> Check the 'regression/result/result.png'



VI. Pipeline mode (Additional)
	In this mode, You can only view the results without saving any images. Cropped images from YOLO are send directly to ResNet. Total Accuracy is printed on Terminal, detailed result is saved on 'classfication/result/matrix.txt'.

	(venv_name) $ cd ../yolo/

	(venv_name) $ python crop.py pipeline        (this command takes a few minutes)

	--> Check the 'classification/result/matrix.txt' file

