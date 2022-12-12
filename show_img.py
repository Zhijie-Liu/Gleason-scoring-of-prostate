import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

if __name__ == '__main__':
    pd_data256 = pd.read_csv('probability/194w_99_108_20201019183332_probability_adjusted256.csv')
    np_data256 = np.array(pd_data256).astype(np.uint8)
    probability = cv2.applyColorMap(np_data256, cv2.COLORMAP_JET)
    probability_299 = cv2.resize(probability, (5400, 4950))
    cv2.imwrite('probability/probability_299.png', probability_299)
    a = 1
