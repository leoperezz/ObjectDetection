# Object Detection




### Make the model

We can find many Checkpoints of the pre-trained models in the tensorflow Object Detection API.

tensorflow object detection API: https://github.com/tensorflow/models/tree/master/research/object_detection

checkpoints: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

Notes:

* It doesn't matter the dimensions of the images, because with model.preprocess() it fits the dimensions of the model.
* The prediction will be a dictionary containing several results that will be processed by model.postprocess().
