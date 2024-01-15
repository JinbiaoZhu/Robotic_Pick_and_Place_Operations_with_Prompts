"""
Description: The abstract class about basic tools of a robot.
Editor: Jinbiao Zhu
Date: 19-10-2024
"""
import datetime
import numpy as np
import matplotlib.pyplot as plt


def get_current_date_and_time() -> str:
    now = datetime.datetime.now()

    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)
    second = str(now.second).zfill(2)

    return year + month + day + hour + minute + second


def observe_mask(mask: np.array):
    plt.figure()
    plt.imshow(mask)
    plt.show()
    plt.close()

