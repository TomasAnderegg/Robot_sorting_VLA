'''
In order to mask the unwanted pill for the robot, I wanted to make sure the robot's 
observation input was already masked. I monkey pathed the "get_observation" so I can 
replace the obs["front"] with the masked version. Avoiding the direct modification of the 
LeRobot library while doing what I wanted. Also allowed me to explore and write my first 
monkey patch.
'''

from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from pill_masking import frame_mask, colour_boxes
import sys
import numpy as np
from typing import Any
from lerobot.scripts.lerobot_rollout import main as lerobot_main
import cv2
from ultralytics import YOLO
import pill_masking                      
from collections import Counter
import os 

import time

RUN_ID = time.strftime("%Y%m%d_%H%M%S")    # e.g. "20261109_134012", computed ONCE when the script starts

frame_count = 0             # frames seen by YOLO in this episode
frames_with_pill = 0        # frames where YOLO found at least one pill
frames_target_seen = 0      # ... where at least one box has the target class (raw YOLO)
frames_target_locked = 0    # ... where the locked target (after votes) is visible = what SmolVLA gets
frames_target_twice = 0     # ... where 2+ boxes have the target class: a confusion (only 1 target pill)
class_frames = Counter()    # in how many frames each class was detected
TASK_PILL = None            # e.g. "amox", set in main()

SAVE_EVERY = 10
SAVE_DIR = r"C:\Users\tjga9\Documents\Tomas\Projets\LeRobot\yolo_dagger_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

TASK_COLOUR = None
PILL_TO_COLOUR = {"amox": "red", "cal": "blue", "nast": "yellow", "ibu": "green"}
TASK_FLAGS = ("--task=", "--dataset.single_task=")   

original_get_observation = ThreadSafeRobot.get_observation #returns a dict with the states and obs

def yolo_mask(img, result, target, pad=7):
    '''Paint over every detected pill that is not the target, using the background colour around it.'''
    H, W = img.shape[:2]
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)

    any_pill = np.zeros((H, W), bool)
    target_mask = np.zeros((H, W), bool)
    distractors = []

    for (x1, y1, x2, y2), c in zip(boxes, classes):
        m = np.zeros((H, W), bool)
        m[max(y1 - pad, 0):y2 + pad, max(x1 - pad, 0):x2 + pad] = True   # box + small margin
        any_pill |= m
        if result.names[c] == target:
            target_mask |= m
        else:
            distractors.append(m)

    mask_img = img.copy()
    big_kernel = np.ones((9, 9), np.uint8)
    for m in distractors:
        big = cv2.dilate(m.astype(np.uint8), big_kernel).astype(bool)
        ring = big & ~any_pill                       # background only, no pill
        paint = m & ~target_mask                     # never paint over the target
        if paint.any() and ring.any():
            mask_img[paint] = np.median(img[ring], axis=0)

    return mask_img

def get_masked_observation(self) -> dict[str, Any]:
    obs = original_get_observation(self)
    
    image_bgr = cv2.cvtColor(obs["front"], cv2.COLOR_RGB2BGR)
    global frame_count
    
    frame_count += 1
    if frame_count % SAVE_EVERY == 0:
        filename = f"{RUN_ID}_{TASK_COLOUR}_{frame_count:06d}.jpg"
        cv2.imwrite(os.path.join(SAVE_DIR, filename), image_bgr)
    result = model.track(image_bgr, conf=0.25, agnostic_nms=True, persist=True, verbose=False)[0]
    coloured_img = colour_boxes(obs["front"], result, target_colour=TASK_COLOUR)
    if result.boxes.id is not None:      # at least one pill tracked in this frame
        count_detections(result)

    obs["front"] = frame_mask(coloured_img, TASK_COLOUR,simu=False)                               
    return obs
        
#Monkey patch    
ThreadSafeRobot.get_observation = get_masked_observation

# model = YOLO(r"C:\Users\tjga9\Documents\Tomas\Projets\LeRobot\yolo_runs\yolo11n_medoc_v2\weights\best.pt")
model = YOLO(r"C:\Users\tjga9\Documents\Tomas\Projets\LeRobot\yolo_runs\yolo11s_medoc_v3\weights\best.pt")

def count_detections(result):
    '''What YOLO saw in this frame. Only called when there is at least one pill.'''
    global frames_with_pill, frames_target_seen, frames_target_locked, frames_target_twice
    names = [result.names[c] for c in result.boxes.cls.cpu().numpy().astype(int)]
    ids = result.boxes.id.cpu().numpy().astype(int)

    frames_with_pill += 1
    class_frames.update(set(names))                  # once per frame per class
    n_target = names.count(TASK_PILL)
    if n_target >= 1:
        frames_target_seen += 1
    if n_target >= 2:
        frames_target_twice += 1
    if pill_masking.TARGET_ID is not None and pill_masking.TARGET_ID in ids:
        frames_target_locked += 1


def print_detection_stats():
    '''End of the episode: how well YOLO followed the target pill.'''
    print(f"\n===== YOLO stats, target '{TASK_PILL}': {frame_count} frames, {frames_with_pill} with a pill")
    if frames_with_pill == 0:
        return
    for label, n in [("target seen (raw YOLO)", frames_target_seen),
                     ("target locked (votes)", frames_target_locked),
                     ("target on 2+ pills", frames_target_twice)]:
        print(f"{label:24s}: {n}/{frames_with_pill} = {100 * n / frames_with_pill:.0f}%")
    print("frames per class        :", dict(class_frames))



def detect_colour_from_task(argument):
    for element in argument:
        if element.startswith(TASK_FLAGS):          # was: "--task="
            words = element.split()
            pills = [pill for pill in PILL_TO_COLOUR if pill in words]
            if len(pills) == 0:
                raise ValueError(f"missing pill in task: {element!r}")
            if len(pills) > 1:
                raise ValueError(f"too many pills in task: {element!r}")
            return PILL_TO_COLOUR[pills[0]]
    raise ValueError("missing task")                # back at the function's level, after the loop




def main():
    
    global TASK_COLOUR
    TASK_COLOUR = detect_colour_from_task(sys.argv)
    for i, arg in enumerate(sys.argv):
        if arg.startswith(TASK_FLAGS):              # was: "--task="
            for pill, colour in PILL_TO_COLOUR.items():
                sys.argv[i] = sys.argv[i].replace(f" {pill} ", f" {colour} ")
            print("Task sent to SmolVLA:", sys.argv[i])

    # lerobot_main()
    global TASK_PILL
    TASK_PILL = [pill for pill, colour in PILL_TO_COLOUR.items() if colour == TASK_COLOUR][0]
    try:
        lerobot_main()
    finally:
        print_detection_stats()      # printed even after Ctrl+C or a crash

            

if __name__ == "__main__":
  
    main()

