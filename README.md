# anomaly-detection-in-proctored-videos
to_do

Description of files

get_test_images.py: downloads a set of images from a given url and stores them in --images_dir
image_demo.py: calculates the posenets for the images in --images_dir and stores the posenet images in --output_dir

videotoframes.py: converts videos in --video_dir and stores every 3rd frame in --frames_dir/(rel_path_of_video in --video_dir)
framestoposenets.py: calculates posenets for frames in every folder of --frames_dir and stores the corresponding posenet images in --posenet_dir and coordinates in --coords_dir

posenets/ : contains utility functions
models/ : contains already built model (mobilenet_v1_101.pth used here)
images/ : set of test images got after running get_test_images
posenet/ : corresponding set of image posenets
coordinates_images/ : corresponding set of coordinates for each pose
