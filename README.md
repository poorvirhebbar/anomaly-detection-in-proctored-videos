(Semi) Automated anomaly detection using Machine Learning can aid proctoring and serve as a deterrent to malpractices. etc.  To this end, we have volunteered to design a machine learning model and an associated process that can be used for anomaly detection as part of remote proctoring. The model has been currently trained by a suitable adaptation of a publicly available action recognition dataset from NTU:  http://rose1.ntu.edu.sg/datasets/actionrecognition.asp 

The code and approach are based on Anomaly Detection in Surveillance Videos, by Sukalyan Bhakat, Ganesh Ramakrishnam,  Best demo paper, In Proceedings of the ACM India Joint International Conference on Data Science and Management of Data, CoDS-COMAD '19, Kolkota, India. 

More details on our work in Video Analytics can be found at https://www.cse.iitb.ac.in/~ganesh/videosurvellianceanalytics

For CS387 project students, the runthrough instructions of the code can be seen in [Guidelines for CS387 project students](./Guidelines_for_CS387_project students.md)


# anomaly-detection-in-proctored-videos
to_do

Description of files

get_test_images.py: downloads a set of images from a given url and stores them in --images_dir

image_demo.py: calculates the posenets for the images in --images_dir and stores the posenet images in --output_dir

Run through:

videotoframes.py: converts videos in --video_dir and stores every 3rd frame in --frames_dir/(rel_path_of_video in --video_dir)

framestoposenets.py: calculates posenets for frames in every folder of --frames_dir and stores the corresponding posenet images in --posenet_dir and coordinates in --coords_dir

posenets/ : contains utility functions
models/ : contains already built model (mobilenet_v1_101.pth used here)
images/ : set of test images got after running get_test_images
posenet/ : corresponding set of image posenets
coordinates_images/ : corresponding set of coordinates for each pose

