import cv2
import  math
import numpy as np
from sklearn.cluster import AgglomerativeClustering

#define size of video frame.
W = 1280
H = 720
left_bottom, right_bottom 	= [W*0.1, H], [W*0.9, H]
left_top, right_top			= [W*0.196, H*0.35], [W*0.7, H*0.35]

#only detect on white and yellow color, other colors will be got rid of.
def filter_white_yellow(image):
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # white color mask
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([10, 0, 100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=mask)


def create_roi(img):
    mask = np.zeros_like(img)
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])
    return cv2.bitwise_and(img, mask)

def BoudingLineValidation(line1, line2):
    if int(line2[0]) in range(min(line1[1], line1[3]), max(line1[1], line1[3])) or \
            int(line2[1]) in range(min(line1[1], line1[3]), max(line1[1], line1[3])):
        return 1  # covered
    return 0  # uncovered

def LaneMagicEngine(image, lines_list):
    if len(lines_list) < 1:
        # print("Data were not enough to compute")
        return -1, 0
    height, width, _ = image.shape
    roi_height = height * 0.9 - right_top[1] * 1.2
    angular_list = []
    for l in lines_list:
        angular_list.append(round(np.rad2deg(np.arctan2(l[3] - l[1], l[2] - l[0])), 0))
    data = np.array(angular_list)
    a = data.reshape((len(angular_list), 1))
    if len(angular_list) == 1: # when HougLine detects one line only
        lane_height = max(lines_list[0][1], lines_list[0][3]) - min(lines_list[0][1], lines_list[0][3])
        if lane_height / roi_height < 0.4:
            l = lines_list[0]
            return True, 0
    cl_num = 1
    std_dev = np.std(a) 
    if std_dev >= 4 and std_dev < 6: 
        cl_num = 2
    elif std_dev >= 6:
        cl_num = 3
    # Apply hierarchical clustering using Agglomerative method(bottom-up)
    cluster = AgglomerativeClustering(n_clusters=cl_num, affinity='euclidean', linkage='ward')
    try:
        cluster.fit_predict(a)
    except:
        print("Could not cluster")
        return -1, 0
    x1, y1, x2, y2 = 0, 0, 0, 0
    if angular_list[0] > 0:
        x1, y1 = right_top[0], right_top[1]
        x2, y2  = right_bottom[0], right_bottom[1]
    else:
        x1, y1 = left_top[0], left_top[1]
        x2, y2  = left_bottom[0], left_bottom[1]

    # this code to calculate average of distance, slope of each group
	# allocation must be critical
    dist_list, coords, means_list  = [None] * cl_num, [None] * cl_num, [None] * cl_num
    groups_line = [[], [], []]
    mydict = {i: np.where(cluster.labels_ == i)[0] for i in range(cl_num)}
    for i in range(len(mydict)):
        y_min = height
        y_max, sum, sum_dist = 0,0,0
        for ix in mydict[i]:
            l = lines_list[ix]
            groups_line[i].append(l)
            y_max = max(y_max, max(l[1], l[3]))
            y_min = min(y_min, min(l[1], l[3]))
            agl = np.rad2deg(np.arctan2(l[3] - l[1], l[2] - l[0]))
            sum += abs(agl)
            # compute distance
            x0, y0 = (l[0] + l[2]) / 2, (l[1] + l[3]) / 2
            dist = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))
            sum_dist += dist
        means_list[i] = sum / len(mydict[i])
        coords[i] = [y_max, y_min]
        dist_list[i] = sum_dist / len(mydict[i])
    # select group with highest slope average
    max_value = max(means_list)
    cl_index  = means_list.index(max_value)
    dist_min_value  = min(dist_list)
    dist_index      = dist_list.index(dist_min_value)
    if dist_index == cl_index and len(mydict) != 1:
        if cl_num > 2 :
            means_list.pop(cl_index)
            groups_line.pop(cl_index)
            max_value = max(means_list)
            cl_index  = means_list.index(max_value)
    RIGHT_COLOR = [120, 18, 254]
    LEFT_COLOR = [253, 108, 46]
    for l in groups_line[cl_index]:
        if angular_list[0] > 0:
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), RIGHT_COLOR, 2, cv2.LINE_AA)
        else:
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), RIGHT_COLOR, 2, cv2.LINE_AA)

    final_gap = False
    lane_height = abs(coords[cl_index][1] - coords[cl_index][0])
    roi_bound = right_top[1] * 1.2
    first_gap = False
    base_line = None

    min_pos = min(coords[cl_index][1], coords[cl_index][0])
    max_pos = max(coords[cl_index][1], coords[cl_index][0])
    if lane_height / roi_height > 0.5:
        # do window sliding
        min_line = 10
        sliding_start = max_pos if max_pos/height > 0.8 else  int(height*0.9 - 5)
        curP = sliding_start
        sliding_boundary = roi_bound if roi_bound / min_pos < 0.18 else roi_bound
        gap_count = 0
        while curP > int(sliding_boundary):
            sliding = [curP, curP - min_line]
            step_gap = False
            for l in groups_line[cl_index]:
                if BoudingLineValidation(l, sliding) == 1:
                    if first_gap == False:
                        base_line = l
                        first_gap = True
                    step_gap = True
            if step_gap is False:
                final_gap = True
                gap_count += 1
            curP -= 5

        if gap_count < 10:
            final_gap = False
    else:
        final_gap = True
        
    return final_gap

def OnStepProcessing(image):
    white_yellow = filter_white_yellow(image)
    gray = cv2.cvtColor(white_yellow, cv2.COLOR_RGB2GRAY)
    smooth_gray = cv2.GaussianBlur(gray, (15, 15), 0)
    aoi_img = create_roi(smooth_gray)
    edges = cv2.Canny(aoi_img, 50, 150)
	# this combination of parameters is quite good, in case of different data, please estimate others parameters
    lines = cv2.HoughLinesP(edges, rho=2, theta=np.pi / 180, threshold=80, minLineLength=10, maxLineGap=10)
    left_list = []
    right_list = []
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            if l[2] == l[0]:
                continue  # ignore a vertical line
            angle = round(np.rad2deg(np.arctan2(l[3] - l[1], l[2] - l[0])), 0)
			#You might calibrate your camera if necessary
            if (angle < 75 and angle > 25) or (angle < -25 and angle > - 75):
                if (angle < 75 and angle > 25):  # belong to right side
                    x1, y1 = right_top[0], right_top[1]
                    x2, y2 = right_bottom[0], right_bottom[1]
                    x0, y0 = (l[0] + l[2]) / 2, (l[1] + l[3]) / 2
                    dist = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / np.sqrt(
                        np.square(x2 - x1) + np.square(y2 - y1))
                    if dist > 5:
                        right_list.append(l)
                if (angle < -25 and angle > - 75):  # left side
                    x1, y1 = left_top[0], left_top[1]
                    x2, y2 = left_bottom[0], left_bottom[1]
                    x0, y0 = (l[0] + l[2]) / 2, (l[1] + l[3]) / 2
                    dist = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / np.sqrt(
                        np.square(x2 - x1) + np.square(y2 - y1))
                    if dist > 5:
                        left_list.append(l)

    if len(left_list) < 2 and len(right_list) < 2:
        return -1

    x1, y1 = 400, 100
    right_gap, left_gap  = -1, -1 # detect lane for each side then process 
    right_gap  = LaneMagicEngine(image,  right_list)
    left_gap   = LaneMagicEngine(image, left_list)

    lane_position = 0
    if  right_gap == -1 and left_gap == -1:
        cv2.imshow("anh", image)
        lane_position = -1
    if  right_gap == False: # RIGHT
        lane_position = 2
    if left_gap == False: # RIGHT
        lane_position = 0
    if left_gap == True and right_gap == True:# MIDDLE
        lane_position = 1
    if left_gap == True and right_gap == -1: # MIDDLE
        lane_position = 1
    if right_gap == True and left_gap == -1:# MIDDLE
        lane_position = 1

    #Drawing postion's text
    TEXT_COLOR = (250, 15, 250)
    if lane_position == 2:
        cv2.putText(image, str("RIGHT"), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                3, TEXT_COLOR, 5)
    elif lane_position == 0:
        cv2.putText(image, str("LEFT"), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                    3, TEXT_COLOR, 5)
    elif lane_position == 1:
        cv2.putText(image, str("MIDDLE"), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                    3, TEXT_COLOR, 5)
    elif lane_position == 3:
        cv2.putText(image, str("LEFT_DOT"), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                    3, TEXT_COLOR, 5)
        lane_position = 0
    elif lane_position == 4:
        cv2.putText(image, str("RIGHT_DOT"), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                    3, TEXT_COLOR, 5)
        lane_position = 2
    else:
        cv2.putText(image, str("Could not detect"), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                    3, TEXT_COLOR, 5)
    cv2.imshow("anh", image)

def main():
    video = cv2.VideoCapture("input.mp4") # place you file path here to process
    while True:
        ret, image = video.read()
        if ret is False:
            break
        OnStepProcessing(image)
        key = cv2.waitKey(25)
        if key == 27:
            break
    video.release()

if __name__ == "__main__":

    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

