from config import Config
from utils.detector_num import Detector as Detector_num
import cv2
import glob
import time

config = Config()
confidence = 0.15
conf_str = str(confidence).replace(".","_")
detectorObj_num = Detector_num()
detectorObj_num.setup("Custom AI", config.PATH_TO_CKPT, config.PATH_TO_LABELS, 2, confidence)


print("detector loaded")

folder = "/content/drive/MyDrive/code-repo/data/test-images"

ctr = 0
tot_time = 0
for filename in glob.glob(folder + "/*"):
    print(filename)
    image = cv2.imread(filename)
    
    boundingBoxImage, filtered_detections, filtered_boxes , time_taken = detectorObj_num.getInferenceResults(image.copy())
    print(f"Time taken is {time_taken}")
    tot_time += time_taken

    cv2.imwrite("/content/drive/MyDrive/code-repo/outputImages/" + str(ctr) + ".jpg", boundingBoxImage)
    ctr += 1
    print(filtered_boxes)

    if ctr == 10:
        break


