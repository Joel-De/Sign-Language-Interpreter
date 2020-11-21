# Sign-Language-Interpreter
Analyses input from a webcam and outputs what letter you are showing in sign language

![Sign language](https://user-images.githubusercontent.com/71618484/93732209-548aab80-fb9e-11ea-8a6b-ed99ac30c0e3.gif)


You can use the Data Collect script to collect a data set of images that you can then train, change the file directory, image size and image count to your own preferences, when run the program will take repeatedly take pictures while testing to make sure they aren't blurry, this value can be changed but is pre-set to 5(works well for a solid color background).

Trainer.py can be used to train the model, there are some variables that you may need to change in order to suit your own needs, most of these can be found at the start of the file. The trainer is currently optimized for my specific model and dataset, you'll probably have to fine tune some values before you can use it.

Determine imports a model from a file directory, this is used to make predictions from your webcam, you will have to change where your model outputs to on your computer as it loads it from a directory

