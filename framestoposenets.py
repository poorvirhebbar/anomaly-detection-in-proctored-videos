import cv2
import time
import argparse
import os
import torch
import sys

import posenet


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--frames_dir', type=str, default='./frames')
parser.add_argument('--posenet_dir', type=str, default='./frame_posenets')
parser.add_argument('--coords_dir', type=str, default='./coordinates')
args = parser.parse_args()

frame_folders = [
    f.path for f in os.scandir(args.frames_dir) if not f.is_file()]

if args.posenet_dir:
    if not os.path.exists(args.posenet_dir):
        os.makedirs(args.posenet_dir)

if args.coords_dir:
    if not os.path.exists(args.coords_dir):
        os.makedirs(args.coords_dir)

def posenet_folder(input_folder, output_folder, coordinate_file):
    model = posenet.load_model(args.model)
    # model = model.cuda()
    output_stride = model.output_stride

    filenames = [
        f.path for f in os.scandir(input_folder) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

    start = time.time()
    for f in filenames:
        input_image, draw_image, output_scale = posenet.read_imgfile(
            f, scale_factor=args.scale_factor, output_stride=output_stride)

        with torch.no_grad():
            input_image = torch.Tensor(input_image)#.cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

        keypoint_coords *= output_scale

        if args.posenet_dir:
            draw_image = posenet.draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.25)

            cv2.imwrite(os.path.join(output_folder, os.path.relpath(f, input_folder)), draw_image)

        if not args.notxt:

            original_stdout = sys.stdout
            with open(coordinate_file, 'a') as f1:
                sys.stdout = f1
                print()
                print("Results for image: %s" % f)
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break
                    print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
            
                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

                sys.stdout = original_stdout


    print('Average FPS:', len(filenames) / (time.time() - start))



for folder in frame_folders:
    posenet_folder_path=os.path.join(args.posenet_dir, os.path.relpath(folder, args.frames_dir))
    if not os.path.exists(posenet_folder_path):
        os.makedirs(posenet_folder_path)

    coordinate_file=str(os.path.join(args.coords_dir, os.path.relpath(folder, args.frames_dir)) + ".txt")
    posenet_folder(folder, posenet_folder_path, coordinate_file)

