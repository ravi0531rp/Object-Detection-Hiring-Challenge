from config import Config
from utils.detector_num import Detector as Detector_num
import cv2
import glob
import time

config = Config()
confidence = 0.15
conf_str = str(confidence).replace(".","_")
detectorObj_num = Detector_num()
detectorObj_num.setup("Custom AI", config.PATH_TO_CKPT_chute_jam, config.PATH_TO_LABELS_chute_jam, 2, confidence)


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



# avg_time = tot_time/ctr
# print(f"Tot time is {tot_time}")
# print(f"Avg time is {avg_time}")
# print(f"Avg fps is {1/avg_time}")

# fname = "9L.mp4"

# cap = cv2.VideoCapture(r"images/"+ fname)
# ctr = 0

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
   
# size = (frame_width, frame_height)

# out = cv2.VideoWriter('images/'+ fname.split(".")[0]+"_"+conf_str +"_out.avi", 
#                          cv2.VideoWriter_fourcc(*'MJPG'),
#                          10, size)
    
# fCount = 0
# while cap.isOpened():
# 	for _ in range(10):
# 		ret, frame = cap.read()
	
# 	if ret:
# 		fCount += 10
		
# 		boundingBoxImage, filtered_detections, filtered_boxes , _ = detectorObj_num.getInferenceResults(frame.copy())
# 		print(len(filtered_boxes))
# 		#cv2.imshow("frame",boundingBoxImage)
# 		#out.write(boundingBoxImage)
# 		#if cv2.waitKey(25) & 0xFF == ord('q'):
# 		#	break
# 	else:
# 		break

# cap.release()
# out.release()
# cv2.destroyAllWindows()

