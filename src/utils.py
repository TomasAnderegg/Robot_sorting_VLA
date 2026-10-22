import math
import numpy as np

#correct order in my data + Mujoco's order
SHOULDER_PAN_IDX = 0
SHOULDER_LIFT_IDX = 1
ELBOW_FLEX_IDX = 2
WRIST_FLEX_IDX = 3
WRIST_ROLL_IDX = 4
GRIPPER_IDX = 5

#Value actuators min max from mujoco so101.xml
SHOULDER_PAN_MIN_VAL =-1.91986 
SHOULDER_PAN_MAX_VAL =1.91986
SHOULDER_LIFT_MIN_VAL = -1.74533 
SHOULDER_LIFT_MAX_VAL = 1.74533
ELBOW_FLEX_MIN_VAL = -1.69 
ELBOW_FLEX_MAX_VAL = 1.69
WRIST_FLEX_MIN_VAL = -1.65806 
WRIST_FLEX_MAX_VAL = 1.65806
WRIST_ROLL_MIN_VAL = -2.74385 
WRIST_ROLL_MAX_VAL = 2.84121
# joints = [SHOULDER_PAN_MIN_VAL, SHOULDER_PAN_MAX_VAL, SHOULDER_LIFT_MIN_VAL, SHOULDER_LIFT_MAX_VAL, ELBOW_FLEX_MIN_VAL, ELBOW_FLEX_MAX_VAL, WRIST_FLEX_MIN_VAL, WRIST_FLEX_MAX_VAL]
GRIPPER_MIN_VAL =-0.17453 
GRIPPER_MAX_VAL =1.74533

PAN_OFFSET_DEG  = 6.4 #degree

SHOULDER_LIFT_OFFSET = 2.9 #degree

def deg_to_rad(deg):
    return deg * math.pi / 180.0

def rad_to_deg(rad):
    return rad * 180.0 /math.pi 

def smolva_to_mujoco_gripper(x, opposite_of_rad=False):
    if opposite_of_rad:
        deg_x = rad_to_deg(x)
        val_x =  (deg_x + 10)/1.3306
        
        return val_x            
    else: 
        val_x_deg = 1.3306 * x - 10
        val_x_rad = deg_to_rad(val_x_deg)
        val_x = np.clip(val_x_rad, GRIPPER_MIN_VAL, GRIPPER_MAX_VAL)
        return val_x

def sanity_check(idx, x):
    '''
    Values get clipped for Mujoco
    '''
    if idx == SHOULDER_PAN_IDX and (x < SHOULDER_PAN_MIN_VAL or x > SHOULDER_PAN_MAX_VAL):
        # print("Attention, out of Mujoco's bound for shoulder pan")
        x = np.clip(x,SHOULDER_PAN_MIN_VAL, SHOULDER_PAN_MAX_VAL )

    if idx == SHOULDER_LIFT_IDX and (x < SHOULDER_LIFT_MIN_VAL or x > SHOULDER_LIFT_MAX_VAL):
        # print("Attention, out of Mujoco's bound for shoulder lift")
        x = np.clip(x,SHOULDER_LIFT_MIN_VAL, SHOULDER_LIFT_MAX_VAL)

    if idx == ELBOW_FLEX_IDX and (x < ELBOW_FLEX_MIN_VAL or x > ELBOW_FLEX_MAX_VAL):
        # print("Attention, out of Mujoco's bound for elbow flex")
        x = np.clip(x,ELBOW_FLEX_MIN_VAL, ELBOW_FLEX_MAX_VAL )

    if idx == WRIST_FLEX_IDX and (x < WRIST_FLEX_MIN_VAL or x > WRIST_FLEX_MAX_VAL):
        # print("Attention, out of Mujoco's bound for wrist flex")
        x = np.clip(x, WRIST_FLEX_MIN_VAL, WRIST_FLEX_MAX_VAL)
    
    if idx == WRIST_ROLL_IDX and (x < WRIST_ROLL_MIN_VAL or x > WRIST_ROLL_MAX_VAL):
        # print("Attention, out of Mujoco's bound for wrist roll")
        x = np.clip(x,WRIST_ROLL_MIN_VAL,WRIST_ROLL_MAX_VAL)

    return x


def conversion(actions_stack, opposite_of_rad= False):
    '''
    Convert values into Mujoco or vice versa
    '''    
    print("action value:", actions_stack[0])
    # print (actions_stack.shape)
    row, column = actions_stack.shape

    action_to_mujoco = np.zeros((row, column))
    for i in range(row):
        for j in range(column):
            if j == GRIPPER_IDX:
                action_to_mujoco[i][j]= smolva_to_mujoco_gripper(actions_stack[i][j], opposite_of_rad=opposite_of_rad)
                # sanity_check(j, action_to_mujoco[i][j])
            elif j == SHOULDER_PAN_IDX:
                if opposite_of_rad:
                    action_to_mujoco[i][j] = rad_to_deg(actions_stack[i][j])+ PAN_OFFSET_DEG
                else:
                    action_to_mujoco[i][j] = deg_to_rad(actions_stack[i][j]- PAN_OFFSET_DEG)
                    action_to_mujoco[i][j] = sanity_check(j, action_to_mujoco[i][j])
            elif j == SHOULDER_LIFT_IDX:
                if opposite_of_rad:
                    action_to_mujoco[i][j] = rad_to_deg(actions_stack[i][j]) + SHOULDER_LIFT_OFFSET
                else:
                    action_to_mujoco[i][j] = deg_to_rad(actions_stack[i][j] - SHOULDER_LIFT_OFFSET)
                    action_to_mujoco[i][j] = sanity_check(j, action_to_mujoco[i][j])
            else: 
                if opposite_of_rad:
                    action_to_mujoco[i][j] = rad_to_deg(actions_stack[i][j])
                else:
                    action_to_mujoco[i][j] = deg_to_rad(actions_stack[i][j]) 
                    action_to_mujoco[i][j] = sanity_check(j, action_to_mujoco[i][j])
                
    
    return action_to_mujoco 