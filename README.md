# mic_dachuang-utilities
Utilities developed to tackle specific problems in one medical imaging "College student Innovation" program (Dachuang). 
## Classification
### cls_proto.py
Testing different classifiers for determination of clue cells.
### testimg.py & testimg_plot.py
Test images and write clue cells detection results to file.
## Making datasets
### Slice.py & cuto1024.py 
Slice images of various resolution to 1024*1024px images, preserving bounding boxes that mostly fit into the picture.
### voc2coco.py 
Convert PASCAL-VOC style annotations to MS COCO format.
### multiple_json2coco.py 
Merge discrete JSON annotations into one, as MS COCO dataset.
### demo.py 
Test a model and draw both generated and pre-annotated bounding boxes on testing images.
## testall.py 
Test all epochs of a run with one config, and compare results of them.
## Other utilities
### notify_with_email.py
Supervise the training progress in background, report at each epoch's start with email, also automatically test the last epoch and send the result with email when a new epoch starts.
