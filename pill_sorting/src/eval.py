from transfer import transfer_to_mujoco
import torch
import numpy as np
from lerobot.datasets import LeRobotDataset
from utils import smolva_to_mujoco_gripper, rad_to_deg, deg_to_rad, conversion
import time
import mujoco
import mujoco.viewer
import mediapy as media
import matplotlib.pyplot as plt
from random_pill_pos import random_pill_positions

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs import PreTrainedConfig
from peft import PeftModel
from lerobot.policies.factory import make_pre_post_processors

ROLLOUT = 1
FOLDED_POSITION = (0.03329552, -1.74533, 1.69, 1.30343497, -0.11124077, -0.14770727)
NOT_FOLDED_POS = (0.03329552, -1.31, 0.77, 1.09,-1.29, 0.0 )
RANDOM_PILLS = True     # True: random layout; False: keep the positions from the XML/keyframe
SEED = 10

SQUARE_POSE = [ 0.3545, -0.06675, 0.0085 ]
TRIANGLE_POSE = [ 0.3545, -0.04025, 0.0085 ]
CIRCLE_POSE = [0.3545, 0.01275, 0.0085 ]
CROSS_POSE = [ 0.3545, -0.01375, 0.0085 ]

EVAL_PROMPTS = { "Put the red pill in the square compartment.": SQUARE_POSE,
                "Put the blue pill in the cross compartment.": CROSS_POSE,
                "Put the yellow pill in the circle compartment.": CIRCLE_POSE,
                "Put the green pill in the triangle compartment.": TRIANGLE_POSE
                }

if __name__ == "__main__":

    # fine_tuning_ckpt = "TomasAnderegg/pill_sorting_post_trained_a100"
    # fine_tuning_ckpt = "TomasAnderegg/pill_sorting_full_ft_a100"
    fine_tuning_ckpt = "TomasAnderegg/pill_sorting_expert_only_a100"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_lora = False 

    cfg = PreTrainedConfig.from_pretrained(fine_tuning_ckpt)
    cfg.device = device
    if is_lora:
        base = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", config=cfg)

        policy = PeftModel.from_pretrained(base, fine_tuning_ckpt).merge_and_unload()

    else:
        policy = SmolVLAPolicy.from_pretrained(fine_tuning_ckpt, config=cfg)

    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=fine_tuning_ckpt, preprocessor_overrides={"device_processor": {"device": device}})


    for i in range(ROLLOUT):
        task = "Put the red pill in the circle compartment." #"Put the blue pill in the cross compartment."

        policy.reset()

        # initialize the Mujoco env
        model = mujoco.MjModel.from_xml_path(r"/home/tjga98/mujoco_menagerie/robotstudio_so101/scene_pills.xml")
        data = mujoco.MjData(model)

        cam_id = model.camera("wrist_cam").id
        # print(model.cam_fovy[cam_id], model.cam_sensorsize[cam_id])   # expect 61.0 and [0. 0.]
        
        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        mujoco.mj_resetData(model,data)

        # Pose init 
        # Robot initial pose :
        data.qpos[:6] = NOT_FOLDED_POS#FOLDED_POSITION # to set initial robot pose; [:6] corresponds to the robot joint values
        data.ctrl = NOT_FOLDED_POS#FOLDED_POSITION # to set initial robot pose
       
        #Pill random pose:
        if RANDOM_PILLS:
            rng = np.random.default_rng(i)
            layout = random_pill_positions(model, data, rng)
            # print("Pill layout:", layout)

        done = False

        done = transfer_to_mujoco(model=model, data=data, policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,task = task)
    