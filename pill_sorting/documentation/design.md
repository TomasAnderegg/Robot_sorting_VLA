# Robotic Medication Sorting using SmolVLA

## Context: 

For over 10 years, I have lived with my disabled aunt. She cannot talk or walk properly and she thinks like a child. She needs extra care and my mom looks after her. Nurses sometimes come to our house to help my mom with basic tasks such as washing and feeding my aunt and preparing her medication in dedicated boxes. However, they often make mistakes and put the wrong medication in the wrong box, which is a serious problem since my aunt is epileptic.

This gave me an idea: why not building a robotic system that helps my mom prepare the medication boxes ?

## Problem breakdown:

First, we need to understand the boxes. Each box represents one day and has 4 compartments: morning, noon, evening and night. Every day, my aunt has to take pills but not at the same times of the day: on Mondays she takes 2 pills in the morning and 1 at night, on Tuesday only at noon etc. This schedule is fixed and does not change from one week to the next.

The difficulty is that, even for the nurses, putting the pills in the right compartments can be tricky because the pills are hard to tell apart. For this project, I will start simple and then gradually add complexity (or at least mention what will be future implementations): the pills will have different colours so that thex can be distinguished. 

Finally, once each box is filled with the correct pills for its day, all the boxes must be stacked in the correct order, from Monday to Sunday. 

Before stating, I need to define: 
1. The dimensions of the containers and of the gripper (including the wrist's position error).
2. The model: SmolVLA because it is adapted for SO101 arm, efficient with low cost hardware and open source.
3. The dataset to create(define one episode e.g. is it going to be a full sorting day ?) recorded with an egocentric wrist camera while teleoperating with my leader arm.
4. The resource I have available
5. Post training process with the dataset
6. Learn to use Mujoco and implement the scene in Mujoco to finally run the evaluation on it.
7. Establish the evaluation protocol 

## Outline of the project
### Research questions
1. How can my robot help my mom to autonomously sorts some medications based on a prescription into a pill organizer for a single day ?
2. How post-training a model affects tasks performance using home made dataset ?

### Assumptions
For this project:

1. All pills have different colours so that they are clearly visible.
2. All pills are considered solid and compact. 
3. All pills are considered to be Lego at first for easy manipulation and visual recognition. It is a good proxie with dimensions: 3.1cm length, 1.5cm width and 1cm height. Easy for the SO101 arm to pick and compatible with the compartement dimensions. 

### Evaluation protocol
The success rate will be measured over N rollouts in Mujoco. One rollout is 1 episode of the trained policy: Mujoco places the 4 coloured pills at random positions inside the specific 20x20cm area, the instruction is given to the robot and the policy outputs the joint values that perform the task. At the end of a rollout, the scene is reset and the pills are placed at new random position again. For each rollout, I check whether the task was completed within the time limit. The test set is designed so that the robot has never seen its tasks during training.

In short:
- 1 rollout = 1 episode: 1 attempt by the robot with a duration of max 45s
- There will be approaximatelly 320 rollouts: 20 rollouts per cell for all 16 cells (12 trained and 4 tests).

I will identify 3 types of failures: 
1. Wrong colour
2. Wrong cell (or compartement)
3. Dropped or missed


## Simulation with Mujoco:

Mujoco is a physics engine that reads a description of a world (environment) from an XML file. In this file, we describe the obects: their shape, color, mass, position etc. Mujoco will apply the physic: makes the objects fall, collide or get pushed. 

The difference objects to simulate:
1. My SO101 arm (exists already: https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)
2. The table where the arm is
3. Pill organizer 
4. Pills of different colours 
5. Camera on the wrist of the arm (included on the SO101 arm XML file)

The Mujoco code follows the tutorial in the Mujoco Python documentation and its Jupyter notebook [1,7].

## Resources and Data flow 
My PC does not have enough compute to post-train a model, I will use Google Colab for post-training. While simulation and evaluation run on my PC. The best way I found is to go through the Hugging Face Hub: push the recorded demos to HF, load them in Colab, post-train, push the checkpoint back to HF and run Mujoco locally.
Corresponded file on GitHub: 
- post_train.ipynb (Colab): load the dataset and the VLA model from HF, post-trains the model and pushes the checkpoint to HF.
- eval.py: loads the checkpoint from HF and the angle conversion functions from utils.py, then evaluates the post-trained model on the test set, converting the joint values.

## Sim to real calibration:
### Joint Mapping:

The pose of the simulated arm must match the real one. I read the real joint values live (via a LeRobot command line), convert them to Mujoco's unit (degrees to radians) and implement a transform that maps the real joint values to Mujoco's, including the corrected offsets.


Follower : Range recorded during LeRobot calibration (in ticks):

| Name          |  Min |  Pos |  Max |
|---------------|-----:|-----:|-----:|
| shoulder_pan  |  688 | 1988 | 3235 |
| shoulder_lift |  828 |  844 | 3237 |
| elbow_flex    |  904 | 3107 | 3115 |
| wrist_flex    |  847 | 2875 | 3198 |
| gripper       | 2016 | 2046 | 3530 |

Leader: Range recorded during LeRobot calibration (in ticks): 

| Name          |  Min |  Pos |  Max |
|---------------|-----:|-----:|-----:|
| shoulder_pan  |  788 | 2096 | 3357 |
| shoulder_lift |  820 |  839 | 3214 |
| elbow_flex    |  942 | 3157 | 3166 |
| wrist_flex    |  782 | 2527 | 3109 |
| gripper       | 2032 | 2046 | 3244 |

Mujoco: Joint ranges (in radians):

| Name          |  Min (rad) | Pos |  Max (rad) |
|---------------|-----------:|:---:|-----------:|
| shoulder_pan  |   -1.91986 |  –  |    1.91986 |
| shoulder_lift |   -1.74533 |  –  |    1.74533 |
| elbow_flex    |   -1.69    |  –  |    1.69    |
| wrist_flex    |   -1.65806 |  –  |    1.65806 |
| gripper       |   -0.17453 |  –  |    1.74533 |

Follower vs Mujoco: Angle spans (in degrees):

| Name          | Real span (°) | MuJoCo span (°) | Gap (°) |
|---------------|--------------:|----------------:|--------:|
| shoulder_pan  |        223.85 |          220.00 |    3.85 |
| shoulder_lift |        211.72 |          200.00 |   11.72 |
| elbow_flex    |        194.32 |          193.66 |    0.66 |
| wrist_flex    |        206.63 |          190.00 |   16.63 |
| gripper       |        133.06 |          110.00 |   23.06 |

The simulated arm must faithfully follow the real followers arm, so I take the measured values and map them to Mujoco's. The motors use 12 bits encoders, so 360° corresponds to 4096 ticks. The gripper opens by ~133°.

The tranform function from Hardware to LeRobot software (in percentage of the joint span):

$$
f(x) = \frac{x - \mathrm{val}_{\min}}{\mathrm{val}_{\max} - \mathrm{val}_{\min}} \times 100\%
$$

NB: This formula only applies to the gripper, the other joints use degrees.


The tranform function from LeRobot software to Mujoco (gripper):

$$
\begin{aligned}
g_d(x) &= 1.3306x - 10 \; (^\circ)
\qquad
g_r(x) &= 0.02322x - 0.174 \; (\mathrm{rad})
\end{aligned}
$$

g_d(x) = 1.3306*x - 10 (°) 
g_r(x) = 0.02322*x - 0.174  (rad) 

The result is then clipped to Mujoco's range. Since the real gripper spans 133°, the Mujoco ranges goes from -10° (lower bound) to $-10° + 133° \approx 123°$. The mapping is therefore the line through the two points (0,-10°) where the closed gripper (0%) is shifted by -10° in the simulation and (100,123.06°). 

NB: 
- In the SO101 XML, the actuator ranges are in radians but LeRobot's config uses degrees. During calibration, I must make sure that Mujoco's values match those of my hardware.
- MJCF files read angles in degrees by default, but this one uses radians because of `<compiler angle="radian" meshdir="assets" autolimits="true"/>`

To check for a measurement bias between the simulation and the real hardware, I put the robot in the same pose twice and compared the real jont values with Mujoco's:


| Name          | Real, try 1 (°) | Real, try 2 (°) | MuJoCo (°) | \|try 2 − try 1\| (°) |
|---------------|----------------:|----------------:|-----------:|----------------------:|
| shoulder_pan  |           -2.95 |            0.75 |          0 |                   3.7 |
| shoulder_lift |           -1.93 |           -0.35 |          0 |                   1.6 |
| elbow_flex    |            6.68 |            3.34 |          0 |                   3.3 |
| wrist_flex    |            2.77 |            4.00 |          0 |                   1.2 |
| wrist_roll    |           -0.22 |            3.56 |          0 |                   3.8 |

Based on those observations, I concluded that there is no significant bias between the real robot and Mujoco except for the gripper. During the hardware calibration (recording the min and max of each joint), the closed gripper is set to 0, whereas in Mujoco a gripper value of 0 does not correspond to fully closed but slightly open which creates an offset. Reproducing the Mujoco pose twice on the real robot, I read 9.57% and 9.41% for the gripper, both corresponding to 0° in Mujoco. The formula $g_{d}(x=9.41) \approx 2.5°$ which is negligible: it is the smaller than the largest try to try difference in the table (~4°).

Sign check: startinf from pose A, I moved each arm joint on the real robot, then matched it in the Mujoco viewer. The sim value had the same sign for all 5 joints.

Checks:
- Compare the recorded camera video with the Mujoco replay side by side. If the poses differ at the same moment, my mapping (unit, sign or gripper formula) is wrong for that joint.
- Compare every converted value with its joint's 'ctrlrange'. If many values are outside, either the mapping is wrong or the real arm went beyond the simulation's limits (possible for wrist_flex and the gripper, given the span table).

## Dataset protocol:
### Setup:

1. Scene: The robot stands on a black table. In front of it, a 20x20 cm area marked with white tape and the same environment is reproduced in Mujoco. The pills are always placed inside this fixed area: the robot should (hopefully) succeed when a pill is inside it and will mostly fail when it is outside.

2. Camera: the only camera is the robot's egocentric wrist camera. 

3. Start: the robot starts in the folded pose. The pills are placed in a random order inside the area, they may be next to each other but never on top of each other.

4. End: an episode is done when the task is completed i.e., the pill is in the right compartment.

5. Instruction: always "Put the {colour} pill in the {compartment} compartment."

6. Duration: about 30s of motion plus 15s to place the pill i.e., at least 45s per episode. The dataset must have enough variety for the policy to learn: the SmolVLA documentation mentions ~50 episodes for a single task when fine-tuning. 

7. Pilot: run a small pilot test first.

Compartment labels: the camer quality is too low to read the text labels (morning, noon, evening, night) so I replaced them with symbols which are much easier to recognise on camera: morning -> circle, noon-> cross, evening -> triangle and night -> square.

#### Train and test cells:
Each (colour, compartment) combination is a cell. For each colour, one cell is held out as a test cell: this combination never appears during training so that testing on it measures the model's understanding. This gives 12 train cells and 4 test cells (the instructions are listed in the Appendix).

#### Recording plan:
- For each train cell (e.g., "Put the red pill in the circle compartment."), I recorded 8 episodes, each with the pill at a random position in the area. This gives 3 x 8 = 24 episodes per colour and with 4 colours, 4 x 24 = 96 training episodes. At 45s per episode, the dataset lasts about 1h12.
- Episodes are grouped into rounds. One round = 4 episodes, one per pill. At start of a round, the 4 pills are placed randomly in the area and the robot performs the 4 tasks. At the end of a round, no pill should be left on the table: all should be in their compartments. The pills are then shuffled and placed randomly in the area again.
- Rounds are balanced: each round uses one train cell per colour, chosen so that after 24 rounds (=94/4), every train cell has appeared exactly 8 times. A round last 4 x 45s = 3min, so the 24 training rounds take 1h12.
- The order of the 4 episodes within a round is random, so that the robot does not learn a rule about the order of the pills.
- A compartment can hold up to 4 pills at the same time which increase data diversity: $3^4 = 81$ possible rounds.

## Post-training:

Post training uses LoRa (via PEFT) instead of classic full fine tuning, to avoid destroying the model's pretrained knowledge. LoRa reduces the number of trainable parameters with rank 64, only 2.97M parameters are trained out of SmolVLA's 453M.

I first used a T4 GPU with batch size 32 and no mixed precision. A 100 step test gave an average of 5.5s per step. To find the bottleneck, I compared the time spent loading a batch on the CPU ('data_s') with the time spent on the forward/backward pass on the GPU ('updt_s'). With this GPU, I only had access to 2 CPU's. On average, I measured 'data_s' = 5.15s and 'updt_s' = 4.89s. To reduce these numbers, I enabled mixed precision (fp16) following the SmolVLA setup and reduced 'num_workers' to 2 (default is 4). I then measured 'data_s' = 0.0004s and 'updt_s' = 1.0s, i.e., 1.03s per step. The SmolVLA authors fine tune for a bit less than 20'000 steps which in my case would have taken just under 5h40. I chose 10'000 steps so that the model sees the data 4.8 times (4.8 epoochs).

I then upgraded to an A100 with high RAM (12 CPUs). This let me match SmolVLA's batch size of 64, increase 'num_workers' to 10 and switch mixed precision to bf16. The whole training took ~54min.


## Results:
### Simulation Scene:

To reproduce the scene as faithfully as possible, I needed to locate the robot's main (base) axis. I then took several measurments to map the scene into my simulation along with the exact sizes of the objects: pill organizer, pills, taped area and distance to the robot.

<p align="center">
  <img src="../assets/scene_measurment.jpeg" width="500">
  <img src="../assets/scene_mujoco.png" width="500">
</p>

A function places the pills at random position inside the area at each episode. I tested the simulation by replaying episode 0, i.e., by sending Mujoco the actions recorded during that episode. The task is to pick the red pill and put it in the circle compartment. The replay reproduced the recorded movement and placement closely, not perfectly so the simulation was ready.

### Correcing the sim to real offsets:
#### Pill explosion diagnosing:

Implementing the geometries from the real measurements of the setup led to the following rendering in Mujoco: 

<p align="center">
  <img src="../assets/so101_and_scene_but_not_yet_there_disappear_2.gif" alt="Offset-Free Tracking Demo" width="80%">
</p>
gif: so101_and_scene_but_not_yet_there_disappear_2.gif


The red pill suddenly vanished. This comes from the simulator's physics: the gripper presses the pill against the compartment and the resulting force launches the pill. Printing the red pill's position over the whole episode and focusing on the "explosion" shows that it jumps from around [0.2 -0.08 0.03] m to  [9.19, 0.41, 1.14] m. This reveals two position errors: the gripper does not reach the right compartment (in episode 0, the robot has to put the red pill in the circle compartment) and the gripper is too low (z-axis) so it touches the compartment and makes the pill explode.

I rewatched the real recording at this exact moment to check whether my gripper was that low and whether the pill touched the compartment. It did not which confirms the position errors: in simulation, the gripper was off by a few centimetres. 

First, I visualised the gripper's contact point (a Mujoco site) to check whether the author of the model file had placed it at the tip of the gripper: 

<p align="center">
  <img src="../assets/so101_and_scene_but_not_yet_there_4.gif" alt="Offset-Free Tracking Demo" width="80%">
</p>
gif: so101_and_scene_but_not_yet_there_4.gif


The contact point's position is acceptable so the error probably comes from the commands my script sends to Mujoco.

##### xy plane: shoulder_pan offset:
To check whether the problem comes from a calibration error, I compared real and simulated positions. I selected a frame of the recording where the gripper is exactly above the circle compartment:

<p align="center">
  <img src="../assets/so101_above_circle_comp.png" alt="Offset-Free Tracking Demo" width="80%">
</p>

Assuming the Mujoco scene maps the real one correctly, the circle symbol is at [0.3545  0.01275 0.0085 ] m so the real gripper is at x = 0.3545m and y = 0.01275m. To get the simulated position, I first extracted the index of that frame with $ idx = fps * t_{s}$ where $t_{s}$ is the time in the recording at which the gripper is at the reference position and $fps = 30$ Hz. I took the follower arm's joint values at that frame, converted them to radians and passed them to Mujoco's kinematics function to get the position Mujoco would have produced. The simulated position is [ 0.33657445 -0.03025805  0.0109119] with x = 0.33657445m and y = -0.03025805m. This confirms a mismatch between the real and simulated values so the calibration needs to be corrected.

I focused first on the gripper's position in the xy plane and computed the angle error between the real and simulated positions:
$$
\theta_{error} = \theta_{real} - \theta_{sim} = 7.18°
$$

where $\theta_{real} = arctan(0.0127/0.3545) = 2.05°$ and $\theta_{sim} = arctan(-0.0302/0.3365) = -5.13°$.

To check whether the error propagates with distance, I repeated the experiment with the square compartment, thr farthest one. I selected the grame of episode 30 where the gripper is exactly above the square symbol and proceeding the same way, I obtained: 

$$
\theta_{error} = \theta_{real} - \theta_{sim} = 5.77°
$$

where $\theta_{real} = arctan(-0.0667/0.3540) = -10.56°$ and $\theta_{sim} = arctan(-0.0978/0.3310) = -16.33°$.

The error has the same order of magnitude as for the circle. Next, I had to find which joint was responsible. My initial visual calibration had already highlighted a few potentially problematic joints and the one that drives motion in the xy plance is shoulder_pan. I substracted from it an offset equal to the mean of the errors (for the circle and square), which makes the simulation match reality, as the video shows:

<p align="center">
  <img src="../assets/so101_problem_xy_plan_solved_offset.gif" alt="Offset-Free Tracking Demo" width="80%">
</p>
gif: so101_problem_xy_plan_solved_offset.gif

##### z-axis: shoulder_lift offset:

The video still shoes the problem on the z-axis which causes the pill explosion that does not happen in the real recording. I used the same process to find the offset to apply to a specific joint. As a reference, I took a frame of episode 0 where the gripper tip touches the table so I can easily say that the gripper is at z = 0m at that frame:

<p align="center">
  <img src="../assets/so101_gripper_z_reference_irl.png" alt="Offset-Free Tracking Demo" width="80%">
</p>

From the index of that frame, I extracted the robot's joint values, converted them and applied forward kinematics to get the position Mujoco would have produced: $ z_{sim} = -0.8 cm$. Resulting in an error for episode 0:

$$
z_{error} = z_{real} - z_{sim} = 0.8 cm
$$

where $z_{real} = 0 cm$ and $z_{sim} = -0.8 cm$.

Repeating the process for another frame, in episode 30, where the gripper touches the table gives $z_{error} = 2.1 cm$.

This error corresponds to the offset I should apply to one of the joints. Three joints directly affect the gripper's z motion and could cause this error: shoulder lift, elbow flex and wrist flex.

<p align="center">
  <img src="../assets/so101_scheme_z_correction_reasoning.jpeg" alt="Offset-Free Tracking Demo" width="80%">
</p>

Two arguments help narrow down the suspects. First, a z error grows linearly with the lever arm and with the angle error introduced by the joint. As the sketch shows with $ L = R*\theta$, the longer the lever arm $R$, the more likely this joint is to propagate its angle error to the gripper. Second, in the span table of the initial calibration, shoulder_lift and wrist_flex have the largest span gaps: when a real command is out of bounds, it gets clipped in Mujoco, which introduces errors. 

For the same frames as before (episodes 0 and 30), I measured $R$ as the distance from each joint to the gripper tip:


| Joint         | R at the near pickup (ep 0) | R at the far pickup (ep 30) |
|---------------|----------------------------:|----------------------------:|
| shoulder_lift | 0.222 m                     | 0.326 m                     |
| elbow_flex    | 0.250 m                     | 0.238 m                     |
| wrist_flex    | 0.159 m                     | 0.159 m                     |
 
To find the suspect, I computed $\theta = L/R$ for each joint, using the two errors measured in episode 0 and 30 (the two main suspects being shoulder_lift and wrist_flex). If a joint carries an angle error, its two $\theta$ values (near and far pickup) should be very similar: 


| Joint         | θ at the near pickup (ep 0) | θ at the far pickup (ep 30) |
|---------------|----------------------------:|----------------------------:|
| shoulder_lift | 2.1°                        | 3.7°                        |
| elbow_flex    | 1.8°                        | 5.1°                        |
| wrist_flex    | 2.9°                        | 7.6°                        | 

As the table shows, the shoulder lift values are close to each other. However, judging the gripper's position from a video, I cannot be more precise than $\pm 1$cm:

<p align="center">
  <img src="../assets/so101_scheme_uncertainty_reasoning.jpeg" alt="Offset-Free Tracking Demo" width="80%">
</p>

This sketch shows how I computed the angle uncertainty:

$$
\theta_u = \pm \frac{L}{R}, \qquad R = \sqrt{x^2 + y^2}
$$

where $L$ is the position uncertainty and $R$ is the distance from the robot's origin to the point $(x,y)$:

The following table presents the overall results:

| Compartment | x (m) | y (m)  | R (m) | θ_u = ±L / R (L = 1 cm)    |
|-------------|------:|-------:|------:|---------------------------:|
| circle      | 0.354 |  0.013 | 0.354 | ±0.0282 rad (±1.62°)       |
| square      | 0.354 | −0.067 | 0.360 | ±0.0278 rad (±1.59°)       |

The angle uncertainty is $\approx 1.6°$. The difference between the near and far shoulder lift angles lies within this range, which makes shoulder_lift suspect number one. I therefore substracted an offset of -2.9° from this joint, which gives:

<p align="center">
  <img src="../assets/so101_problem_xyz_solved_offset.gif" alt="Offset-Free Tracking Demo" width="80%">
</p>
gif: so101_problem_xyz_solved_offset.gif

### Closed loop evaluation:

<p align="center">
  <img src="../assets/loop.jpeg" width="500">
</p>

As showns in the illustration, the loop has 3 blocks: the environment where the robot is (Mujoco or the real world), the conversion blocks to connect the environment and the VLA correctly and the VLA model itself.

1. Environment: as discussed in the previous section, The Mujoco environment was build to be as faithful as possible to the real setting by respecting the measurements and the physics.
2. Conversion blocks: these were the most challenging part, since they required a good understanding of the input and output of both the environment and the VLA model. They let the two sides of the loop to understand each other's values. Mujoco only works with radians and the VLA with degree so the block from the environment to the VLA converts the joint values from radians to degrees, except for the gripper which needs the mapping function described in the calibration section. The other blocks does the opposite.

3. VLA model: it converts the state and images observations together with the task instruction into actions. It produces 50 actions at once (an action chunk) instead of a single action. A specific build-in function (select_action function) either generates a new chunk of 50 actions or releases the actions of the current chunk at the policy's 30 Hz frequency. This chunk generation allow the model to be called less time making it efficient. For an episode lasting at most 85.3s, 'select_action' is called 2559 times but the VLA generates actions only 52 times. The rest of the time, the action of the current chunk are executed sequentially one every 0.0333s.   

#### Stuck in the folded pose:
I first evaluated the policy on a task the robot had never seen, to assess its understanding but the robot did not move at all from its folded pose. I then switched to a task it had been trained on: "Put the red pill in the circle compartment.". and ran into the same problem:

<p align="center">
  <img src="../assets/lora_first_video_folded_pose.gif" width="500">
</p>
gif: 
lora_first_video_folded_pose.gif

The robot stays in its folded pose and does not move at all. 3 runs gave the same results. I first thought it was a simulation problem so I deployed the policy on my real robot:

<p align="center">
  <img src="../assets/folded_real.gif" width="500">
</p>
gif: folded_real.gif

The problem persists. As a first experiment, I compared the actions produced by the model with the human's actions. I selected 2 distinct frames of episode 0 (F150 and F625), computed the actions with the fine tuned model and compared them with the human's: both match within 2.3°:

| Joint | F150 Policy | F150 Human | Diff | F625 Policy | F625 Human | Diff |
|:------|------------:|-----------:|-----:|------------:|-----------:|-----:|
| Pan (°) | 8.85 | 8.57 | 0.28 | 25.16 | 25.36 | 0.21 |
| Lift (°) | -103.43 | -103.47 | 0.04 | 15.68 | 17.85 | 2.17 |
| Elbow (°) | 97.49 | 97.05 | 0.44 | 6.72 | 7.12 | 0.40 |
| Wrist flex (°) | 50.15 | 52.44 | 2.29 | 63.48 | 61.76 | 1.73 |
| Wrist roll (°) | -104.86 | -106.33 | 1.47 | -82.71 | -82.86 | 0.15 |
| Gripper | 0.97 | 1.16 | 0.19 | 13.10 | 13.37 | 0.27 |

Frame 625 was chosen because the arm is far from the folded pose which makes it a good comparison point.

This rule out a broken model. Analysing the dataset recordings, I then noticed that every episode I make starts and ends the robot in the folded pose, which makes it the most recorded pose in the whole dataset. I therefore made the robot start from a non folded pose by moving it slightly. For the same trained task, the robot then moved better than when trembling in the folded pose, both on the real robot and Mujoco:
 
<p align="center">
  <img src="../assets/not_folded_real_lora.gif" width="400">
  <img src="../assets/lora_first_video_not_folded_pose_gpu.gif" width="400"> 
</p>
gifs: lora_first_video_not_folded_pose_gpu.gif and not_folded_real_lora.gif

These tests show that the robot get stuck in the folded pose whereas starting from a non folded pose lets the policy produce actions. In simulation, the robot finally moves towards the pill to accomplish its task: "Put the red pill in the circle compartment.".

#### Visual understanding:

However, the robot did not succeed in putting the red pill in the circle compartment, so testing an unseen task in pointless. It does grab a pill but not the correct one and puts it in a compartment other than the circle. This shows that the robot has learned the motion of grabbing a pil and placing it in a compartment but struggles to identify the symbols and colours. This is probably due to the poor camera quality during recording which makes the images very blurry and to the camera shaking a lot while the robot moves. In addition, the model was post trained for $\approx 1$h so it saw the data only about 4 times.

To tackle the visual understanding problem, I could either re-record the dataset with a better camer or post train again with a full fine-tuning of all parameters, which also fine tunes the vision encoder on the dataset's frames. The following figure shows the result of full fine tuning on the task:"Put the red pill in the circle compartment.":

<p align="center">
  <img src="../assets/not_folded_real_full_ft.gif" width="500">
</p>
gif: not_folded_real_full_ft.gif

Out of 4 runs, the robot picked the red pill only in the first one. In the following runs with the pills rearranged in different orders but the same task, the robot still struggled to spot the correct pill and to grab it. A final post training run used the same setup as the original paper (fine tuning only the action expert). It gave the same result on the task: "Put the red pill in the circle compartment." and did not improve performance. 

#### Conclusion:
I conclude that three factors explain why the robot did not learn the tasks well even though the model was pretrained: the amount of recorded data is too small, the image quality is poor, since the robot recognised neither the colours nor the symbols and the dataset is not diverse enough as all episodes share the same start and end pose.

## Problems and solutions:
1. Unreadable compartment labels: the camera resolution was too low to read the text on the pill organizer so I replaced the text with symbols to make sure the robot can recognise the compartments
2. frame index error in the dataset: To avoid mistakes during data recording, I wrote a script that automatically runs every task 9 times. Later,during post training, I hit an error about frame count mismatch. I checked the frame count mismatch for all episoded and extracted the problematic ones. The missing frames are always at the end so the inconsistency probably comes from the script encoding a video while already starting the next recording (during recording, each episode ended with the robot folded once the task was completed). There were two solutions: $(i)$ trim the video with an unofficial tool $(ii)$ re-record the problematic episodes.

For simplicity and to avoid introducing new complicated errors, I choose $(ii)$. I re-recorded the corrupted episodes with the plls in exactly the same positions, so as not to break the diversity of pill positions. I then deleted the corrupted episodes from the original dataset and merged the new episodes into it.

3. Colab time limit: During post training on the T4 GPU, my colab session disconnected because of the time limit. Upgrading the GPU allowed me to shorten the post training.

4. Noisy replay and elbow_flex tracking gap:
For the simulation, I used the SO101 XML provided by Google DeepMind. When loading the first action of episode 0 to check that joint order was correct, I noticed a noisy movement. Printing the joint calues sent to Mujoco showed that some of them were out of bounds, so I clipped them before sending them to Mujoco but the problem remained. Since Mujoco moves the joint with a PD controller (qpos = the robot's position in the simulation, ctrl = the target position), I checked for every joint whether the target was reached. Joint 2 (elbow flex) shows a gap of ~17° at the beginngin which indicated a collision between geometric parts that creates a repulsive force. Since this does not affect the purpose of the project, it is not a priority:

<p align="center">
  <img src="../assets/replay_first_frames_ep0_tracking.png" width="500">


  <img src="../assets/so101_tweaking_2_action0.gif" alt="Offset-Free Tracking Demo" width="80%">
</p>
gif: so101_tweaking_2_action0.gif

## Questions for myself
Q1: The scale problem. Do the numbers.

R1: I will need to measure the pill, compartment and the SO101 gripper maximum opening in mm.
The pill compartment measures: 10.8 cm de lenght, 4.5 cm de width, 1.5 cm wall height. Each compartement is separated with a distance of 2.5 cm. The pill size: 2 cm de lenght, 0.5 cm de width et 0.5 cm wall height. The max opening of the gripper: 15 cm.

Q2: The grasp. Watch yourself pick up a pill. How many fingers do you use, and how? Then compare that with a two-finger
  parallel gripper.
  → What does your retargeting turn a fingertip pinch into? Is "gripper closed" when thumb and index finger touch a good
  enough rule for a 7 mm object?

R2: Since I will use a leader arm to teleoperate and grab the pill, I won't need to worry about fingers.

Q3: The simulator. Does Mujoco or LIBERO contain a pill organizer, or anything close to one?
  → If not, what would you have to build, and how hard is it in MuJoCo with simple shapes (cylinders, a grid of boxes)? Does
  that change your answer to Q2 from last time (LIBERO vs. a simpler sim)?

R3: There is no pill organizer to the best of my knowledge. I will need to construct it. The easiest thing to do is using Mujoco and build it there. I have always wanted to try Mujoco Simulator. Hopefully it won't need to much time, I am aiming at spending no more than a day and half.

Q4: What is one episode? "Sort a whole week" means 7 days × 2–4 times × several pills: a very long horizon.
  → What's the smallest unit you'd record and evaluate? And how does the robot know what to do: one instruction per pill, or
  a whole "prescription"?

R4: Not sure how hard this project is going to be for only a week, I will keep it simple so 1 episode represents 1 pill -> right container at the right side. And also for simplicity, I will start by only a day of sorting. If I have time, I will add another day. 

Q5: Generalization with a small dataset. Say you have 3 pill types and 7 compartments.
  → How many (pill, compartment) combinations is that? Do you need to record all of them? What would it show if the robot
  succeeded on a combination it never saw during training? (This is likely your answer to Q3 from last time: the thing that
  makes a reviewer read more carefully.)

R5: According to the task description, the composition is fixed by the description during training. The idea would be to create a small dataset with different combination according to the task description: I will start by one pill, different combination then progressively scale up the number of pills up to 4 with different combinations. The idea is to use at inference the robot trained on a zero shot full new description, recognize the pull and able to place it in the right side of the box.

Imagine 3 pills colours (Red, blue, yellow) and 4 compartments (Morning, Noon, Evening and Night). I will record demos only for the train cells for example for red: Morning(train), Noon(train), Evening(train) and Night(test). During test time, I will give an instruction the robot has never seen before "put the red pill in the night compartment". This will allow me to evaluate if the robot hasn't just memorized the motion but understood red and night.

## Annexe
Prompt instruction for training:

Put the red pill in the circle compartment.
Put the red pill in the cross compartment.
Put the red pill in the triangle compartment.
Put the blue pill in the circle compartment.
Put the blue pill in the triangle compartment.
Put the blue pill in the square compartment.
Put the yellow pill in the cross compartment.
Put the yellow pill in the triangle compartment.
Put the yellow pill in the square compartment.
Put the green pill in the circle compartment.
Put the green pill in the cross compartment.
Put the green pill in the square compartment.

Prompt instruction for test:

Put the red pill in the square compartment.
Put the blue pill in the cross compartment.
Put the yellow pill in the circle compartment.
Put the green pill in the triangle compartment.

## References
1. https://mujoco.readthedocs.io/en/stable/python.html
2. https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotstudio_so101/so101.xml
3. https://huggingface.co/docs/lerobot/en/installation
4. https://arxiv.org/abs/2506.01844
5. https://huggingface.co/docs/peft/index
6. https://huggingface.co/blog/samuellimabraz/peft-methods
7. https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb#scrollTo=slhf39lGxvDI

