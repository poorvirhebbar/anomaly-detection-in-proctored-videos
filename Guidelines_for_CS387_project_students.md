**Anonymization:**
Renaming of files done here: https://colab.research.google.com/drive/18DmaHJuTL0qUWbn8kSqOqCA_dEfRz6Od?usp=sharing

**Github Files runthrough:** https://github.com/poorvirhebbar/anomaly-detection-in-proctored-videos

**videotoframes.py:**
converts videos in --video_dir and stores every 3rd frame in --frames_dir/(rel_path_of_video in --video_dir) 
Usage: python3 videotoframes.py --video_dir [video_directory] --frames_dir frames [frames directory]

**framestoposenets.py:**
calculates posenets for frames in every folder of --frames_dir and stores the corresponding posenet images in --posenet_dir and coordinates in --coords_dir
Usage: python3 framestoposenets.py --frames_dir [frames_directory] --posenet_dir [posenet_directory] --coords_dir [pose_coordinates_directory]

A demo can be found here but the videos are not accessible to everyone so this can’t be compiled by others: https://colab.research.google.com/drive/1SPayt4D4JJsRoW44BbeZChh8wssWTKSK?usp=sharing

**Autoencoder part:**
This takes in the coordinates as input nodes and trains on it given the anomalous and non-anomalous labels. The trained autoencoder is then used for testing and the former model framework is https://colab.research.google.com/drive/19PLR-Bj7bBJ9G9gj6KP-Qqswc9uv4TCj?usp=sharing