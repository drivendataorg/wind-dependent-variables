PATH_TO_ROOT = './'
import sys

sys.path.append(PATH_TO_ROOT)


# Base
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import click

import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Specific
from src.data.storms_dataset import get_storms_df, StormsDatasetSequence
from src.utils.misc import multithreaded_map


DATA_DIR = './data'  # Path from project root
OUTPUT_DIR = 'dense_optical_flow'
SAVE_TO = os.path.join(PATH_TO_ROOT, DATA_DIR, OUTPUT_DIR)

def draw_flow(img, flow, grid, reverse=False, arrows=True, color=(0, 255, 0)):
    h, w = img.shape[:2]
    x, y = grid
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = img.copy()
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    new_grid_x, new_grid_y = [], []
    for (x1, y1), (x2, y2) in lines:
        x2 = max(0, min(w-1, x2))
        y2 = max(0, min(h-1, y2))
        if arrows:
            if reverse:
                cv2.circle(vis, (x2, y2), 1, color, -1)
            else:
                cv2.circle(vis, (x1, y1), 1, color, -1)
        new_grid_x.append(x2)
        new_grid_y.append(y2)
    return vis, (np.array(new_grid_x), np.array(new_grid_y))


def process_img(i_img):
    global storms_data_reader

    irow = storms_data_reader.df.iloc[i_img]
    output_filepath = os.path.join(SAVE_TO, irow.image_id + ".jpg")
    img, _, _ = storms_data_reader[i_img]

    ##https://nanonets.com/blog/optical-flow/
    new_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # new_img[..., 0] = img[..., 0]
    # new_img[..., 1] = img[..., int(round((img.shape[2]-1)/2))]
    # new_img[..., 2] = img[..., -1]
    prev_fr = img[..., 0]
    step = 16
    h, w = new_img.shape[:2]
    grid_y, grid_x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    new_grid = (grid_x, grid_y)
    colors = np.linspace(0, 255, img.shape[-1] - 1).astype(np.uint8)
    for iimg in range(1, img.shape[-1]):
        next_fr = img[..., iimg]

        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        flow = cv2.calcOpticalFlowFarneback(prev_fr, next_fr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        color = [int(s1 * 255) for s1 in plt.get_cmap('gist_rainbow')(colors[iimg - 1])[0:3]]
        new_img, new_grid = draw_flow(new_img, flow, new_grid, reverse=True, arrows=iimg == 1, color=color)
        prev_fr = next_fr

    Image.fromarray(new_img).save(output_filepath)


@click.command()
@click.option('--only_test', is_flag=True)
def main(only_test):

    # Get dataset
    dset_df = get_storms_df(os.path.join(PATH_TO_ROOT, DATA_DIR))
    if only_test:
        dset_df = dset_df[dset_df.test]

    # Output directory
    if not os.path.isdir(SAVE_TO):
        os.mkdir(SAVE_TO)

    global storms_data_reader
    storms_data_reader = StormsDatasetSequence(dset_df, 5, gap=0.5)

    max_workers = 16*1

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with tqdm(total=len(storms_data_reader)) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_img, i_img) for i_img in range(len(storms_data_reader))]
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)



if __name__ == '__main__':
    main()