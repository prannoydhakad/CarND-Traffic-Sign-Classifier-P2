#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report




## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/prannoydhakad/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Signs_Recognition.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used basic maths to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

The exploratory visualization of the data set. It is a bar chart showing how the training set is spread across the signs


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I decided to grayscale and normalize the image, to produce A) less uneeded varience, and B to give a mean of 0 to help the optimizer

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Data was already split, I decided not to use augmented data at this time.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


My Final Model layout is as follows:
Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
Layer 2:Pooling. Input = 28x28x6. Output = 14x14x6
Layer 3: Convolutional. Output = 10x10x16
Layer 4: Pooling. Input = 10x10x16. Output = 5x5x16.
Layer 5: Flatten. Input = 5x5x16. Output = 400.
Layer 6: Fully Connected. Input = 400. Output = 120.
Layer 6 - 7 : Dropout layer
Layer 7: Fully Connected. Input = 120. Output = 84.
Layer 7 - 8: Dropout layer
Layer 8: Fully Connected. Input = 84. Output = n_classes.


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used a dropout level of 0.7 and a larning rate of 0.001 , together with an epoch of 200 to get the greatest results so far

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I chose Adam Optimizer in the end, after experimenting with Adadelta and finding it seemed to not quite work for my needs.
Convolutional layers, were useful thanks to the fact the sign could sometimes be in different parts of the image, it let the net be more general.
I mostly used guesswork at the start for the hyperparameter, but as it got closer, i moved in smaller steps, kind of like a manual schotastic descent.

My final model results were:
* validation set accuracy of 0.947
* test set accuracy of 0.925
I chose LeNet, as it seemed to be simple and well suited for this problem, I beleive there may have been others available, which if given extra time, would research.  

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Image 1: 
The blue sky behind the sign makes it harder to distinguish, similar with image 5

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model got 4 out of 5 of the signs correct.
this 80% accuracy is worse then the test set, but I suspect with more new traffic signs, that would settle closer to the training/validation set's 93% accuracy.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The probabilities are pretty low for my model, a sign I have overfit, or chosen the wrong archecture.