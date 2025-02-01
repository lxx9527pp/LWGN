import torch
from models.test import LWGN
import matplotlib.pyplot as plt
import os
import glob
import cv2 as cv
from models.common import post_process_output
import numpy as np
from utils.dataset_processing.evaluation import detect_grasps

def plot_output(rgb_img, grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None):

    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    fig = plt.figure(figsize=(10, 10))
    fig1 = plt.figure(figsize=(10, 10))
    ax = fig1.add_subplot(1, 1, 1)
    ax.imshow(rgb_img)
    idx = 0
    for g in gs:
        g.plot(ax)
        idx += 1
        if idx == no_grasps:
            break
    ax.axis('off')

    ax = fig.add_subplot(3, 1, 3)
    plot = ax.imshow(grasp_width_img, cmap='jet')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(3, 1, 1)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(3, 1, 2)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.axis('off')
    plt.colorbar(plot)
    plt.show()


model_path = 'C:/Users/11316/Desktop/实验/deep_neet/ggcnn-master/pt/epoch_25_iou_0.94_statedict.pt'

file_path = 'E:\jacquard_dataset'

if __name__ == '__main__':
    # model = SwinTransformerSys(in_chans=3, embed_dim=48, num_heads=[1, 2, 4, 8]).to('cuda')
    model = LWGN(in_channels=3).to('cuda')
    model.load_state_dict(torch.load(model_path))
    # rgbf = glob.glob(os.path.join(file_path,  '*r.png'))
    rgbf = glob.glob(os.path.join(file_path, '*', '*', '*_RGB.png'))
    rgbf.sort()
    rgbf = rgbf[int(len(rgbf)*0.9):]
    # rgbdf = [f.replace('perfect_depth.tiff', 'RGB.png') for f in rgbf]
    rgbdf = [f.replace('RGB.png', 'perfect_depth.tiff') for f in rgbf]
    idx = 0
    with torch.no_grad():
        for x in rgbf:
            img = cv.imread(x)
            print(x)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (224, 224))
            # img = img[208:432, 128:352]
            i = img
            img = img.transpose(2, 0, 1)
            img = img / 255.0
            img = torch.from_numpy(img).float().unsqueeze(0).to('cuda')
            q, c, s, w = model.forward(img)
            q_out, ang_out, w_out = post_process_output(q, c, s, w)
            # grasps = detect_grasps(q_out, ang_out, w_out)
            plot_output(rgb_img = i, grasp_q_img = q_out, grasp_angle_img = ang_out,  grasp_width_img=w_out)
            print(rgbdf[idx])
            idx += 1