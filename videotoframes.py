import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', type=str, default='./videos')
parser.add_argument('--frames_dir', type=str, default='./frames')
args = parser.parse_args()

if args.frames_dir:
    if not os.path.exists(args.frames_dir):
        os.makedirs(args.frames_dir)

filenames = [
    f.path for f in os.scandir(args.video_dir) if f.is_file() and f.path.endswith(('.mp4'))]



def FrameCapture(path, framepath): 
	if not os.path.exists(framepath):
		os.makedirs(framepath)

	vidObj = cv2.VideoCapture(path) 
	count = 0
	success = 1

	while success: 
		success, image = vidObj.read() 


		if(count%3==0):
			output_path=framepath + "/frame%d.jpg" % (count/3)
			cv2.imwrite(output_path, image)

		count += 1

if __name__ == '__main__': 


	for f in filenames:
		tmp=os.path.relpath(f, args.video_dir)
		fp=os.path.join(args.frames_dir, os.path.splitext(tmp)[0])
		# fp=os.path.join(args.frames_dir, os.path.relpath(f, args.video_dir))

		print(f,fp)
		FrameCapture(f,fp) 
