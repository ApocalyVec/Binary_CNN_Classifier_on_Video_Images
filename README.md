# Binary CNN Classification on Video Images
This program trains a Binary Classifier on classifying images from video files.
The classifier is based on CNN with 3 Convolution layers and two Dense layers.

## Categories
The two categories used in the program call 'on' and 'not'

## Running the program
1. To run the program, put the videos of the two categories in the project root folder.
2. To add videos, change the video list in 212 and 213 to your video names (you can skip this step in the future if you just 
wish to train the classifier, more details in step #3)
3. If you did step #2, set isConvertFrames and isCreateTrainTest to True in line 214 so that the old frame images
will be overwritten. If you skipped step 2, set isConvertFrames and isCreateTrainTest to False so that 
the program won't convert the videos to frames again.
4. run main.py, the classifier will be save in a folder named trained_models once it is trained. The name of the classifier
will be model_<datetime>