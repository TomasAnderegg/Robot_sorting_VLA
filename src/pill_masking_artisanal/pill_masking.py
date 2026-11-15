import numpy as np
import cv2
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from collections import Counter
'''
input: Image (H, W, 3, uint8, RGB) and the task colour
output: masked RGB image except for the target pill.

If you want to test this function: python pill_masking.py

'''
PILL_COLOUR = {"amox": "red", "cal": "blue", "nast": "yellow", "ibu": "green"}

# RGB values chosen so frame_mask's HSV ranges detect them (checked one by one)
PAINT_RGB = {"red": (220, 30, 30), "blue": (30, 125, 220), "yellow": (230, 190, 32), "green": (25, 180, 125)}

history = {}
HISTORY_LEN = 15

TARGET_ID = None
frames_since_seen = 0
LOCK_VOTES = 8        # votes needed to lock a target
LOST_FRAMES = 30      # frames without seeing it before unlocking


# def colour_boxes(img, result):
#     '''Fill every YOLO box with the colour of its pill. img: (H, W, 3) uint8 RGB.'''
#     out = img.copy()
#     boxes = result.boxes.xyxy.cpu().numpy().astype(int)
#     classes = result.boxes.cls.cpu().numpy().astype(int)
#     if result.boxes.id is None:        # tracker found no pill in this frame
#         return img.copy()
#     ids = result.boxes.id.cpu().numpy().astype(int)
#     for (x1, y1, x2, y2), c, track_id in zip(boxes, classes, ids):
#         out[y1:y2, x1:x2] = PAINT_RGB[PILL_COLOUR[result.names[c]]]
#     return out

# def yolo_mask(img, result, target, pad=7):
#     '''Paint over every detected pill that is not the target, using the background colour around it.'''
#     H, W = img.shape[:2]
#     boxes = result.boxes.xyxy.cpu().numpy().astype(int)
#     classes = result.boxes.cls.cpu().numpy().astype(int)

#     any_pill = np.zeros((H, W), bool)
#     target_mask = np.zeros((H, W), bool)
#     distractors = []

#     for (x1, y1, x2, y2), c in zip(boxes, classes):
#         m = np.zeros((H, W), bool)
#         m[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad] = True   # box + small margin
#         any_pill |= m
#         if result.names[c] == target:
#             target_mask |= m
#         else:
#             distractors.append(m)

#     mask_img = img.copy()
#     big_kernel = np.ones((9, 9), np.uint8)
#     for m in distractors:
#         big = cv2.dilate(m.astype(np.uint8), big_kernel).astype(bool)
#         ring = big & ~any_pill                       # background only, no pill
#         paint = m & ~target_mask                     # never paint over the target
#         if paint.any() and ring.any():
#             mask_img[paint] = np.median(img[ring], axis=0)

#     return mask_img


def colour_boxes(img, result, target_colour):
    '''Fill every YOLO box with its pill's colour. Only the most-voted target ID keeps
    the target colour; any other "target" is painted as a distractor so frame_mask
    erases it. The target box is filled last, so an overlapping distractor box can
    never cut it. img: (H, W, 3) uint8 RGB.'''
    global TARGET_ID, frames_since_seen
    out = img.copy()

    if result.boxes.id is None:         # tracker has no pill yet in this frame
        return out

    pill_on = True
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    ids = result.boxes.id.cpu().numpy().astype(int)

    # Stable class and its number of votes for each pill
    colours = []
    counts = []
    for c, track_id in zip(classes, ids):
        name, count = stable_class(track_id, result.names[c])   # ← FIX 1: two values
        if name in PILL_COLOUR:
            colours.append(PILL_COLOUR[name])
        else:
            colours.append(None)       # unknown class: will be skipped
        counts.append(count)           # ← FIX 2: fill counts like colours

    # A colour that is NOT the target, to turn the losers into distractors
    # (must be defined BEFORE step 3 uses it)
    other_colour = None
    for colour in PAINT_RGB:
        if colour != target_colour:
            other_colour = colour
            break

    # Only ONE target allowed: the ID with the most votes
    # 1. Is the locked target still visible?
    if TARGET_ID is not None:
        if TARGET_ID in ids:
            frames_since_seen = 0
        else:
            frames_since_seen += 1
            if frames_since_seen > LOST_FRAMES:
                TARGET_ID = None          # really lost: allow a new election

    # 2. No locked target: elect one by votes
    if TARGET_ID is None:
        best_i = None
        for i in range(len(colours)):
            if colours[i] == target_colour:
                if best_i is None or counts[i] > counts[best_i]:
                    best_i = i
        if best_i is not None and counts[best_i] >= LOCK_VOTES:
            TARGET_ID = ids[best_i]
            frames_since_seen = 0

    # 3. Only the locked ID keeps the target colour
    for i in range(len(colours)):
        if ids[i] == TARGET_ID:
            colours[i] = target_colour                # force it, even if it now looks like "ibu"
        elif colours[i] == target_colour and TARGET_ID in ids:
            colours[i] = other_colour                 # any other "target" becomes a distractor

    # Paint: distractors first, target last
    order = sorted(range(len(boxes)), key=lambda i: colours[i] == target_colour)
    for i in order:
        if colours[i] is None:
            continue
        x1, y1, x2, y2 = boxes[i]
        out[y1:y2, x1:x2] = PAINT_RGB[colours[i]]  # only the pill: keeps its shape and angle
    return out

def frame_mask(img,task, simu):

    '''
    Extracted the HSV range values from different frames from ep0. For each pill colour 
    that does not corresponds to the target-> apply a mask. Created a mask then dilate it
    to make sure it cover any type of colour reflexion. I wanted to apply a mask of the background.
    A mask of the surrounding background from each unwanted pill because it changes for every pill.
    Extracted a contour (=ring) for every unwanted pill that corresponds to the pill's contour background.
    Made sure that it does not mix other pill colours and then apply the median to extract the colour of the 
    mask. Apply it to the RGB image. A final check whether a pill is there or not.
    '''

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    if simu:
        colour_mask = {
            "red":    ((h <= 8) | (h >= 172)) & (s > 150) & (v > 80),   # red wraps around 0/179
            "blue":   (h >= 110) & (h <= 130) & (s > 150) & (v > 80),
            "yellow": (h >= 22) & (h <= 38) & (s > 150) & (v > 80),
            "green":  (h >= 60) & (h <= 82) & (s > 150) & (v > 25),     # your green is dark (V = 69 at most)
        }
    else:
        colour_mask = {
            "red":    ((h <= 13) | (h >= 170)) & (s > 90) & (v > 60),
            "blue":   (h >= 95) & (h <= 115) & (s > 90) & (v > 60),
            "yellow": (h >= 14) & (h <= 34) & (s > 45) & (v > 120),
            "green":  (h >= 75) & (h <= 94) & (s > 70) & (v > 30),
        }
    
    target = colour_mask[task]
    any_pill = colour_mask["red"] | colour_mask["blue"] | colour_mask["yellow"] | colour_mask["green"]

    mask_img = img.copy()
    kernel = np.ones((7, 7), np.uint8)
    big_kernel = np.ones((9, 9), np.uint8)

    for colour, val in colour_mask.items():
        if colour == task:
            continue 

        closed = cv2.morphologyEx(val.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        dil = cv2.dilate(closed, kernel).astype(bool) & ~target
        big = cv2.dilate(dil.astype(np.uint8), big_kernel).astype(bool)
        ring = big & ~dil & ~any_pill

        if dil.any() and ring.any():  #colour may be absent from this frame
            mask_img[dil] = np.median(img[ring], axis=0)

    return mask_img



def stable_class(track_id, class_name):

    # First time we see this pill: create its empty list
    if track_id not in history:
        history[track_id] = []

    # Add the class predicted in this frame
    history[track_id].append(class_name)

    # Keep only the last 15 predictions
    if len(history[track_id]) > HISTORY_LEN:
        history[track_id].pop(0)

    # Count each class and return the most frequent one
    counts = Counter(history[track_id])
    best_class = None
    best_count = 0
    for name, count in counts.items():
        if count > best_count:
            best_class = name
            best_count = count

    return best_class, best_count

if __name__ == "__main__":
    #Testing zone

    # ds = LeRobotDataset("TomasAnderegg/pill_sorting_colour_direct_mix_v1",episodes=[0], download_videos = True)
    ds = LeRobotDataset("TomasAnderegg/so101_pill_sorting_medoc",episodes=[0], download_videos = True)
    
    img = (ds[145]["observation.images.front"].permute(1,2,0).numpy()*255).astype(np.uint8)
    
    masked_img=frame_mask(img, task="yellow")

    # #Yolo model testing 
    # from ultralytics import YOLO
    # model = YOLO(r"C:\Users\tjga9\Documents\Tomas\Projets\LeRobot\yolo_runs\yolo11n_medoc_v1\weights\best.pt")
    # print(model.names)   # should print {0: 'amox', 1: 'cal', 2: 'ibu', 3: 'nast'}; 80 names means the wrong file
    # result = model(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), conf=0.25, verbose=False)[0]
    # print(len(result.boxes), [result.names[int(c)] for c in result.boxes.cls], result.boxes.conf.tolist())
    # # masked_img = yolo_mask(img, result, target="amox")
    # coloured_img = colour_boxes(img, result)
    # masked_img = frame_mask(coloured_img, task=PILL_COLOUR["cal"])

    #VIZZZ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(img); ax1.set_title("RGB"); 
    ax2.imshow(masked_img); ax2.set_title("HSV"); 
    plt.tight_layout()
    plt.show()
   
