"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm-preliminary-development-phase | gzip -c > example-algorithm-preliminary-development-phase.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path
from glob import glob
import os
import numpy as np
import time
import torch
from PIL import Image
import importlib

from core.utils import to_tensors

from helper import *

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

neighbor_stride = 5

# resize frames
def resize_frames(frames, size):
    frames = [f.resize(size) for f in frames]
    return frames, size

# sample reference frames from the whole video
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    ref_length = 10
    for i in range(0, length, ref_length):
        if i not in neighbor_ids:
            ref_index.append(i)
    return ref_index

def run():
    now = time.time()
    # check if cuda is available
    show_torch_cuda_info()

    # Read the input
    input_location = INPUT_PATH / "images/synthetic-surgical-scenes"
    mask_location = INPUT_PATH / "images/synthetic-surgical-scenes-masks"

    # each scene & corresponding mask is a multi-page tiff
    input_files = glob(str(input_location / "*.tiff")) + glob(str(input_location / "*.mha"))
    mask_files = glob(str(mask_location / "*.tiff")) + glob(str(mask_location / "*.mha"))

    print(f"Found {len(input_files)} input files")
    print(f"Found {len(mask_files)} mask files")

    # iterate over all test scenes
    for i, file in enumerate(input_files):
        new_now = time.time()
        with open("/output/test.txt", "w") as f:
            f.write("Time loading image files: " + str(new_now - now) + " s\n")
        now = new_now

        # load the image and corresponding mask
        input_id = get_scene_id(file)
        mask_id = IMAGE_MASK_MAP[input_id]

        print(f"Processing scene {input_id}...")

        input_array = load_image_file_as_array(file)
        mask_array = load_image_file_as_array(os.path.join(mask_location, 
                                                           f"{mask_id}.mha"))
        
        orig_size = np.asarray(input_array).shape[1:3]
        orig_size = (orig_size[1], orig_size[0])
        shape = np.array(np.asarray(input_array).shape)
        
        input_array = [Image.fromarray(frame) for frame in input_array]
        mask_array = [Image.fromarray(np.reshape(mask, mask.shape[:-1])) for mask in mask_array]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        size = (432, 240)

        net = importlib.import_module('model.e2fgvi_hq')
        model = net.InpaintGenerator().to(device)
        data = torch.load("release_model/E2FGVI-HQ-CVPR22.pth", map_location=device)
        model.load_state_dict(data)
        model.eval()

        # prepare data
        frames = input_array
        frames, size = resize_frames(frames, size)
        h, w = size[1], size[0]
        shape[1] , shape[2] = h, w
        video_length = len(frames)
        np_frames = [np.array(f).astype(np.uint8) for f in frames]

        tmasks, size = resize_frames(mask_array, size)
        bmasks = [1 - np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in tmasks]
        with open("/output/test.txt", "a") as f:
            f.write(str(np.bincount(np.reshape(tmasks[0], (-1)))) + "\n")

        sub_len = 48
        num_subvideos = (video_length // sub_len) + 1
        tmasks = [tmasks[sub_len * k : sub_len*(k+1)] for k in range(num_subvideos-1)] + ([tmasks[num_subvideos*sub_len:]])
        bmasks = [bmasks[sub_len * k : sub_len*(k+1)] for k in range(num_subvideos-1)] + ([bmasks[num_subvideos*sub_len:]])
        frames = [frames[sub_len * k : sub_len*(k+1)] for k in range(num_subvideos-1)] + ([frames[num_subvideos*sub_len:]])
        np_frames = [np_frames[sub_len * k : sub_len*(k+1)] for k in range(num_subvideos-1)] + ([np_frames[num_subvideos*sub_len:]])

        new_now = time.time()
        with open("/output/test.txt", "a") as f:
            f.write("Time preparing image arrays: " + str(new_now - now) + " s\n")
        now = new_now

        # completing holes by e2fgvi
        sub_results = []
        iteration_counter = 1
        for sub_tmasks, binary_masks, sub_frames, sub_np_frames in zip(tmasks,
                                                        bmasks,
                                                        frames,
                                                        np_frames):
            if len(sub_tmasks) == 0:
                continue
            imgs = to_tensors()(sub_frames).unsqueeze(0) * 2 - 1
            masks = to_tensors()(sub_tmasks).unsqueeze(0)
            imgs, masks = imgs.to(device), masks.to(device)
            subvideo_length = len(sub_tmasks)

            comp_frames = [None] * subvideo_length

            for f in range(0, subvideo_length, neighbor_stride):
                neighbor_ids = [
                    i for i in range(max(0, f - neighbor_stride),
                                    min(subvideo_length, f + neighbor_stride + 1))
                ]
                ref_ids = get_ref_index(f, neighbor_ids, subvideo_length)
                selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
                selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
                with torch.no_grad():
                    masked_imgs = selected_imgs * selected_masks
                    mod_size_h = 60
                    mod_size_w = 108
                    h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                    w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
                    masked_imgs = torch.cat(
                        [masked_imgs, torch.flip(masked_imgs, [3])],
                        3)[:, :, :, :h + h_pad, :]
                    masked_imgs = torch.cat(
                        [masked_imgs, torch.flip(masked_imgs, [4])],
                        4)[:, :, :, :, :w + w_pad]
                    pred_imgs, _ = model(masked_imgs, len(neighbor_ids))
                    pred_imgs = pred_imgs[:, :, :h, :w]
                    pred_imgs = (pred_imgs + 1) / 2
                    pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
                    for i in range(len(neighbor_ids)):
                        idx = neighbor_ids[i]
                        img = np.array(pred_imgs[i]).astype(
                            np.uint8) * binary_masks[idx] + sub_np_frames[idx] * (
                                1 - binary_masks[idx])
                        if comp_frames[idx] is None:
                            comp_frames[idx] = img
                        else:
                            comp_frames[idx] = comp_frames[idx].astype(
                                np.float32) * 0.5 + img.astype(np.float32) * 0.5

            sub_results = sub_results + comp_frames

            new_now = time.time()
            with open("/output/test.txt", "a") as f:
                f.write(f"Time spent in iteration {iteration_counter}: " + str(new_now - now) + " s\n")
            now = new_now
            iteration_counter += 1

        # Save the output
        sub_results = [Image.fromarray(frame.astype(np.uint8)) for frame in sub_results]
        sub_results, _ = resize_frames(sub_results, orig_size)
        sub_results = np.array([np.array(frame, dtype = np.uint8) for frame in sub_results])

        write_array_as_image_file(
            location=os.path.join(OUTPUT_PATH, "images", 
                                  "inpainted-synthetic-surgical-scenes"),
            scene_id=input_id,    
            array=sub_results,
        )
    
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
