# Sign-Language-Interpreter
Analyses input from a webcam and outputs what letter you are showing in sign language

You can use the Data Collect script to collect a data set of images that you can then train, change the file directory, image size and iamge count to your own prefferences, when run the program will take repeatedly take pictures while testing to make sure they aren't blurry, this value can be changed but is preset to 5(works well for a solid color background).

Trainer.py can be used to train the model, there are sonme variables that you may need to change in order to suit your own needs, most of these can be found at the start of the file

Determine imports a model from a file directory, this is used to make predictions from your webcam

