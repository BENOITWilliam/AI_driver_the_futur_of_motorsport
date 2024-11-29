# 🏎️ Yolo - AI driver the futur of motorsport 🏎️

The objective of this project was to train a vision model that detect FIA (International Automobile Federation) regulation during races. In the futur a AI category of race could appear. This project is aimed to think about how the visions models will be train and the limitation of using it. This work could be important for a company who wants to try to enter in this new categories.

To accomplish this objective I have done this work like if the train vision model will be in a race car in the near futur. For that, I decided to use a fast and lightweight vision model such as Yolov11s. During the race, the vision detection needs to be fast to have the time to modify the car's actions before getting a penalty or a crash. Furthermore, in all the competition you have technical regulations that change according on the category. Having a lightweight vision model allow you to go in all technical regulation and, generally, requires lighter components which is extremely important in racing. This can be important for a company because they will not need to work on a new vision model when the technical regulations of the category change or when they want to enter in a new one.

<a href="https://docs.ultralytics.com/models/yolo11/" target="_blank">
  <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/refs/heads/main/yolo/performance-comparison.png" alt="YOLO11 performance plots">
</a>


## Dataset

The dataset is a mix of two dataset from roboflow. The labeling of this images are also from roboflow where I have change the name of some classes such as car. <a href="https://universe.roboflow.com/project-duycq/f1-vjcba">The first dataset</a> is only composed of flags, <a href="https://universe.roboflow.com/f1detection/detect-cars-irh8v">the second dataset</a> is for car detection.
The dataset is divided with 95% of the images for training, 4% for validation and 1% for the test.


In a first time I started to train my vision model on collab. I tried three version, the nano version that was very quick but had a lot of error so I have abandoned the idea of using it, the small one, with quick and good results and the medium version, who was to long to finish. In the end, because of collabs limitations, I finally train my vision model on a local computer.

All the following pictures come from the last training with Yolov11s :

The last training is a training with the small version using 100 epochs that has taken 1H40 using a Laptop RTX 3050 Ti.
At the beginning of the project, I have search for methods that can make the training less long and more accurate. One of the most used solution is to use a gray and white filter but, in this project we need to know the colors of the différents flags, we can't use filters. In the end the solution to increase the accuracy of the vision model was to give images with noise during the training that could simulate the high speed during a race or some weather condition such as the rain.
