import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def draw_curve(x, y, title, save_path, x_label):
    """
    :param x: array
    :param y: scale (iteration)
    :param title: title of curve
    :param save_path: save path
    :return: none
    """
    x = x.cpu().numpy if isinstance(x, torch.Tensor) else np.array(x)
    y = np.arange(y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.plot(y, x)
    save_path += '/curve'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_path + '/' + title + '.png')
    plt.show()