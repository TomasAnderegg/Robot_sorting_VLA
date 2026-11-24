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
- For each train cell (e.g., "Put the red pill in the circle compartment."), I recorded 25 episodes, each with the pill at a random position in the area. This gives 3 x 25 = 75 episodes per colour and with 4 colours, 4 x 75 = 300 training episodes. At 14.7s per episode, the dataset lasts about 1h13.
- The order of the 4 episodes within a round is random, so that the robot does not learn a rule about the order of the pills. The starting position of the robot has been randomized around the folded pose to avoid the folded pose problem.
- Instead of recording consequently every task at once. I recorded every task consequently so I can a lighting of the environment evenly distributed.
- Recorded episode where a pill is already inside a compartment but not the one on the task.
- Recorded episode to simulate failure and recover from them (DART)

## Post-training:

I replicate the post-training strategy from the SmolVLA paper: post-trained the action expert over 20'000 steps with a dataset with a total of 242137 frames. 

I first used a T4 GPU with batch size 32 and no mixed precision. A 100 step test gave an average of 5.5s per step. To find the bottleneck, I compared the time spent loading a batch on the CPU ('data_s') with the time spent on the forward/backward pass on the GPU ('updt_s'). With this GPU, I only had access to 2 CPU's. On average, I measured 'data_s' = 5.15s and 'updt_s' = 4.89s. To reduce these numbers, I enabled mixed precision (fp16) following the SmolVLA setup and reduced 'num_workers' to 2 (default is 4). I then measured 'data_s' = 0.0004s and 'updt_s' = 1.0s, i.e., 1.03s per step. The SmolVLA authors fine tune for a bit less than 20'000 steps which in my case would have taken just under 5h40. I chose 10'000 steps so that the model sees the data 4.8 times (4.8 epoochs).

I then upgraded to an A100 with high RAM (12 CPUs). This let me match SmolVLA's batch size of 64, increase 'num_workers' to 10 and switch mixed precision to bf16. The whole training took ~54min.


## Results:
### Simulation Scene:
In order to replicate as faithfully as possible the scene, I needed to change the colour of the SO101 robot. In the previous version, the robot was yellow and mine white. This created confusion during the policy deployment both on simulation and on my real hardware. Once the colour changed, a noticable change occured with the new dataset. I tested the new policy trained with the new dataset with the yellow robot against the white robot.


### Closed loop evaluation:

<p align="center">
  <img src="../assets/loop.jpeg" width="500">
</p>

As showns in the illustration, the loop has 3 blocks: the environment where the robot is (Mujoco or the real world), the conversion blocks to connect the environment and the VLA correctly and the VLA model itself.

1. Environment: as discussed in the previous section, The Mujoco environment was build to be as faithful as possible to the real setting by respecting the measurements and the physics.
2. Conversion blocks: these were the most challenging part, since they required a good understanding of the input and output of both the environment and the VLA model. They let the two sides of the loop to understand each other's values. Mujoco only works with radians and the VLA with degree so the block from the environment to the VLA converts the joint values from radians to degrees, except for the gripper which needs the mapping function described in the calibration section. The other blocks does the opposite.

3. VLA model: it converts the state and images observations together with the task instruction into actions. It produces 50 actions at once (an action chunk) instead of a single action. A specific build-in function (select_action function) either generates a new chunk of 50 actions or releases the actions of the current chunk at the policy's 30 Hz frequency. This chunk generation allow the model to be called less time making it efficient. For an episode lasting at most 85.3s, 'select_action' is called 2559 times but the VLA generates actions only 52 times. The rest of the time, the action of the current chunk are executed sequentially one every 0.0333s.   

#### Stuck in the folded pose:
Now with the new dataset (TomasAnderegg/so101_pill_sorting_v3), I made sure to vary a bit the starting pose around the folded pose. This results on an improvment both on simulation and on the real hardware of this problem. Now the robot move to accomplish its task whithout lagging or getting stuck on the folded pose.

#### Visual understanding and Task understanding:
Despite the fact the folded problem is fixed, the policy still struggles to grab the correct pill. After conducting 4 experiment with the task "Put the red pill in the circle compartment.", the robot grabbed the wrong pill all the time but succeed to put it on the correct compartment. This was observed during the deployment of the policy on my real hardware.

However during the deployment on simulation, the robot couldn't do either task. This is probably due to the sim-to-real gap. To improve this, a dataset on simulation will be recorded and the policy will be retrained with both data: from the teleoperation and simulation.

To assess the problem, I needed to test the policy to see if the problem was related to the task description or the visual content understanding. I took an episode selected 2 frames from that episode where the robot sees the environment with all the pills to assess its ability to detect the correct pill colour and go for it. And another frame where the robot has grabbed the pill and it is about to choose the compartment, this allows me to assess whether the robot take into account the task description. 

<p align="center">
  <img src="../assets/idx_colour.png" width="49%">
  <img src="../assets/idx_comp.png" width="49%">
</p>

By loading the post-trained policy, extracting the state observation, task, visual observation and the correct actions produced I was able to spot the problem. Proceed to run the experiment with 3 different episodes and also by fixing the starting noise the action expert is using to only focus on those experiment and not take into account the random noise the action expert is starting with. The following figures only show for episode 150 for different frames:


Task ref: "Put the yellow pill in the cross compartment."\
Task compartment changed: "Put the yellow pill in the square compartment." 
<p align="center">
  <img src="../assets/comp_task_ref.png" width="49%">
  <img src="../assets/comp_task_comp.png" width="49%">
</p>


Task ref: "Put the yellow pill in the cross compartment."\
Task colour changed: "Put the blue pill in the cross compartment."
<p align="center">
  <img src="../assets/colour_task_ref.png" width="49%">
  <img src="../assets/colour_task_colour.png" width="49%">
</p>

If the policy understand the task and understand the colour, the beheviour should change when changing the colour or compartment. By observing the first 2 plots about the compartment changes, we clearly see that the model adjust the joint value to the new compartment compared to the reference.
Whereas for the colour, the policy does not seem to show this behaviour.

To verify this assumption, I ran different ablations, where I tested the model by changing: 
1. Varying the noise with the state values and task unchanged: over 5 different seeds.
2. Changing the colour: "red" instead of "yellow" for episode 150
3. Changing the compartment word
4. A meaningless word of the task: "Place" instead of "Put" 
 
Beforehand, I tested the model's reproducibility by running the policy with the same image, same states, same task and same seed twice resulting in a deterministic ($\Delta = 0$) model. This allows me to avoid introducing noise to the analysis so that every difference in the ablations come from the thing I changed (seed or word), not from hidden randomness in the pipeline.

The first ablation consists of comparing the policy with 5 different seeds. This allows me to determine the variability of the policy under different noise. This corresponds to the reference that I will compare the rest of the ablation with.

The second ablation consists of swapping the colour word of the task with another. This analysis was done 5 times using 5 different seeds. Then I compared the gap between the reference curve with the colour swapped and take the maximum value. This captures whether the model reacts to the colour word. 

Similarly, the third ablation swaps the compartment word of the task by replacing "cross" with "circle". This allows me to detect if the compartment word has an impact on the policy's prediction at this early stage of the episode. 

Finally, the last one consists of swapping the verb word of the task by replacing "Put" by "Place". This ablation allows me to detect if by changing a meaningless word inside the task would affect the policy's prediction. Comparing this result and the colour swap would allow me to see if the colour word is meaningless or not.

#### Results:
Results from episode 150 at the phase where the gripper is closed and empty, approaching the pills, frame 150 (t = 5.0 s), 5 seeds. Frame selected because the model is expected to take its colour decision as soon as it sees the coloured pills.

Reference prompt: *"Put the yellow pill in the cross compartment."*

| Joint | Noise (seed pairs, n=10) | Colour yellow→red (n=5) | Compartment cross→circle (n=5) | Verb Put→Place, control (n=5) |
|---|---|---|---|---|
| shoulder_pan  | 4.60 ± 1.21 | 1.09 ± 0.29 | 1.29 ± 0.13 | 0.79 ± 0.22 |
| shoulder_lift | 7.69 ± 1.62 | **17.53 ± 4.96** | 2.33 ± 1.32 | 4.19 ± 1.43 |
| elbow_flex    | 9.54 ± 2.27 | 10.09 ± 2.84 | 3.82 ± 1.05 | 2.87 ± 1.06 |
| wrist_flex    | 8.06 ± 2.14 | 6.11 ± 0.98 | 1.30 ± 0.16 | 1.33 ± 0.10 |
| wrist_roll    | 5.73 ± 1.52 | 1.44 ± 0.44 | 2.51 ± 0.77 | 1.29 ± 0.23 |
| gripper       | 0.22 ± 0.04 | 0.09 ± 0.01 | 0.12 ± 0.02 | 0.09 ± 0.01 |

*Values: maximum absolute difference over the 50-step action chunk between two predicted trajectories, mean $\pm$ std over pairs. Units: degrees (gripper: % opening). Noise column: same prompt, all 10 pairs of the 5 seeds. Word columns: one word changed, same seed (5 paired comparisons). Determinism check: repeated prediction with identical input and seed gives* $\max |\Delta| = 0$.

<p align="center">
  <img src="../assets/motion_var.png" width="90%">
</p>

The first column compares how the model varies depending on the noise for each joint value. The second column, it compared the colour swap and the reference with the same seed for all joints, similarly for the compartment swap and verb swap ("Place" instead of "Put").

The table suggests that the shoulder lift (17.53 ± 4.96), elbow flex (10.09 ± 2.84) and wrist flex (6.11 ± 0.98) joints are influenced by the colour swap. By analyzing the shoulder lift's value, we notice the interval from the colour swap does not overlap the noise one as well as the compartment and verb one. Suggesting the colour has a greater impact on the policy for this joint. Notice that for the colour swap the noise has a greater impact by looking at the standard deviation.

For elbow_flex, changing the colour word moves the joint about 3.5 times more than the meaningless verb change, with no overlap between the two ranges. However, this effect is about the same size as the variation caused by changing the random seed alone, so it cannot be separated from noise.

Similarly, for the wrist flex colour value, we notice the colour swap has more impact than the compartment and verb swap. But the colour swap values and the noise values overlap suggesting the noise is slightly larger than the colour swap values.

In conclusion, the first observation made earlier suggesting that the colour word does not influence the policy's prediction is wrong, the ablation done shows it is not ignored but its effect is comparable to the sampling noise on most joints. Also notice, this ablation was only done on episode 150, frame 150 on 5 different seeds. The results shows the arm could reach further but not that it will reach the red pill.

I created a counterfactual dataset to get a ground truth for every colour in the same scene. For each layout, I recorded 4 episodes with the same pill positions, the same start pose and the same compartment. Only the colour in the task changes. This let me check whether changing the colour word makes the policy move toward the requested pill, or whether it makes the same movement whatever the colour is.

I tested two checkpoints: one trained without the counterfactual data (full_ft_a100) and one trained with it (colour_cf_departure_mix_ft_v2). For each layout, I took one frame from the layout's first episode and gave the policy the same image, the same robot state and the same noise, changing only the colour word. The policy predicts 50 steps of movement, and I kept the arm position at step 36, using the base rotation, shoulder and elbow angles. I compared it with the positions I recorded when I actually went to each pill: the closest one, by Euclidean distance over these three joint angles, counts as the pill the policy chose. Each prediction adds +1 to the cell (colour asked, pill chosen). With 12 layouts, 4 colours and 5 noise seeds, this gives 240 predictions per checkpoint (Figure 1).

<p align="center">
  <img src="../assets/confusion_matrix_full_ft_a100_chkpt_3joints.png" width="45%">
  <img src="../assets/confusion_matrix_pill_sorting_colour_cf_departure_mix_ft_v2_chkpt_3joints.png" width="45%">
</p>
<p align="center">
  <em><b>Figure 1.</b> Colour-grounding confusion matrices (colour word asked × pill chosen, 240 predictions each), using the base rotation, shoulder and elbow angles at step 36. Chance level is 25%.
  <b>Left:</b> checkpoint trained without counterfactual data (<code>full_ft_a100</code>).
  <b>Right:</b> checkpoint trained with counterfactual data (<code>colour_cf_departure_mix_ft_v2</code>), evaluated on its own training episodes.</em>
</p>

Both checkpoints stay close to chance level (25%). In both matrices, the rows look alike: the arm goes to roughly the same place whatever colour is asked.

To check whether 30% is really above chance, I used the standard error of a proportion:

$$
\sigma = \sqrt{\frac{p\,(1-p)}{n}}
$$

with $p = 0.25$ (chance level). This formula assumes that every prediction is independent. My 240 predictions are not: they come from 48 situations (12 layouts × 4 colours), each repeated with 5 seeds on the same image. I therefore used $n = 48$ rather than $n = 240$, which gives $\sigma = 6.25\%$ (instead of $2.8\%$). The distance from chance is then

$$
z = \frac{30\% - 25\%}{6.25\%} = 0.8
$$

so the result lies well within $\pm 2\sigma$ of chance level. There is no evidence that the policy uses the colour word to choose its target, even though the counterfactual checkpoint was evaluated on episodes it was trained on.

Training on the counterfactual data did not fix this. This matches a known weakness of VLA policies: they grasp reliably but fail to select the instructed object ([Emukpere et al., 2026](https://arxiv.org/abs/2602.24143)). I therefore moved target selection out of the policy: a YOLO detector finds the requested pill and masks the others, so the policy only has to grasp.

One last hypothesis was tested. SmolVLA does not learn from whole episodes but from individual frames, sampled at random: each frame is one training example. The decision about which coloured pill to go to is made at the start of an episode, in roughly the first 90 frames (3 s), while grasping, carrying and dropping the pill take several hundred frames. Every episode has a start, but it is short compared with the rest, so only about 17% of the training examples cover the decision moment. Notice that not all of these 90 frames require the colour word because once the arm starts moving, its position already shows the direction, so the true share of decision frames is probably lower. The rest of the episode teaches the policy nothing about the colour word, because once the arm is heading toward a pill, the images and joint values are enough to finish the task. A wrong decision at the start is therefore hard to recover from: the arm keeps going toward the wrong pill.

To give this moment more weight, I oversampled it. In addition to the counterfactual episodes, I added the first 90 frames of each of them three more times, so the model sees every departure four times (the original plus 3 copies). Departure frames went from about 17% of the training examples to about 44%. This adds no new information; it only makes the training push much harder on "go in the right direction for the colour named".

This checkpoint is the one shown on the right of Figure 1, and it still stays close to chance level: even with the decision moment oversampled, the policy did not learn to follow the colour word.

#### Different post-training method 

I wanted to test whether the policy's use of the colour word could be improved by continuing post-training from the previous checkpoint with the VLM unfrozen. In the expert-only run, only the action expert was trained, including the cross-attention projections through which it reads the VLM's keys and values. Unfreezing the VLM additionally lets the language-model layers adapt to the task which changes how the prompt and the image tokens are encoded before the expert reads them. Because these layers process both the text and the image tokens, this experiment alone cannot tell whether a failure comes from understanding the colour word or from finding the colour in the image.

#### Results

| Joint | Noise (seed pairs, n=10) | Colour yellow→red (n=5) | Compartment cross→circle (n=5) | Verb Put→Place, control (n=5) |
|---|---|---|---|---|
| shoulder_pan  | 4.25 ± 1.08 | 1.27 ± 0.23 | 1.81 ± 0.19 | 0.65 ± 0.10 |
| shoulder_lift | 6.59 ± 1.96 | **11.91 ± 3.24** | 1.43 ± 0.87 | 4.70 ± 1.37 |
| elbow_flex    | 8.35 ± 2.23 | 6.97 ± 2.29 | 2.92 ± 0.94 | 3.60 ± 1.19 |
| wrist_flex    | 7.15 ± 1.88 | 5.72 ± 0.64 | 1.14 ± 0.12 | 1.49 ± 0.13 |
| wrist_roll    | 6.14 ± 1.52 | 2.16 ± 0.99 | 2.89 ± 0.49 | 1.58 ± 0.22 |
| gripper       | 0.20 ± 0.05 | 0.11 ± 0.01 | 0.15 ± 0.02 | 0.09 ± 0.01 |

<p align="center">
  <img src="../assets/motion_var_vlm_unfrozen.png" width="90%">
</p>

The table suggests that the shoulder lift (11.91 ± 3.24), elbow flex (6.97 ± 2.29) and wrist flex (5.72 ± 0.64) joints are influenced by the colour swap. By analyzing the shoulder lift's value, we notice the interval from the colour swap does not overlap the noise one as well as the compartment and verb one. Suggesting the colour has a real impact on the policy for this joint even for this new post-training method. Notice that for the colour swap the noise has a strong impact by looking at the standard deviation.

For elbow_flex, changing the colour word moves the joint about twice as much as the meaningless verb change and more than the compartment change, although its range slightly overlaps with the verb's. However, this effect is smaller than the variation caused by changing the random seed alone, so it cannot be separated from noise.

For the wrist flex colour value, we notice the colour swap has more impact than the compartment and verb swap. But the colour swap values and the noise values overlap suggesting the noise is slightly larger than the colour swap values.

Now by comparing this method with the previous post-training method, by looking at the different analysis made for the 3 joint values, the result got slightly worse according to the table below. Concluding that this method did not help the model to understand better the colour word.

**Colour-word sensitivity relative to noise and to the control: frozen VLM (`pill_sorting_expert_only_A100_big_data`) vs unfrozen VLM (`pill_sorting_vlm_unfrozen_A100`), episode 150, frame 150, 5 seeds**

| Joint | colour ÷ noise (frozen) | colour ÷ noise (unfrozen) | colour ÷ verb (frozen) | colour ÷ verb (unfrozen) |
|---|---|---|---|---|
| shoulder_lift | 2.28 | 1.81 | 4.18 | 2.53 |
| elbow_flex    | 1.06 | 0.83 | 3.52 | 1.94 |
| wrist_flex    | 0.76 | 0.80 | 4.59 | 3.84 |

*Ratio of mean max-over-time |Δ| values from the two result tables. A ratio close to 1 means the colour word is hard to tell apart from seed noise (left columns) or from a meaningless rewording (right columns).*

No further experiments were done to solve this open problem. 

#### Conclusion:

Despite three experiments, the model struggles to detect the correct colour. [Emukpere et al., 2026] also mentioned this problem in their experiments. To carry on this project, as mentioned earlier, a YOLO model will be used to detect the correct pill according to the task and will mask the others. Since the model knows how to grasp, the YOLO model and the masking will allow the project to work.

### Visual Model

The previous tests showed that the policy does not reliably use the colour word to choose its target. Instead of asking the policy to make that choice, I moved it into a separate perception module: the camera image is edited at every frame so that, ideally, only the requested pill remains visible, and the policy only has to grasp.

I first built the masking with coloured Lego bricks (red, blue, yellow, green) standing in for pills. The function `frame_mask` detects each colour with HSV ranges that I measured on frames from a recorded episode. For every colour that is not the target, it builds a mask of that colour, closes small holes, and dilates it slightly to cover edges and reflections, without touching the target. It then takes a band of background around the masked regions (the "ring": a larger dilation minus the masked regions and minus any other brick) and paints the masked regions with the median colour of that band. The median is computed once per colour, so all bricks of one colour become one flat patch of background colour. The patches are still visible, but they no longer look like bricks.

<p align="center">
  <img src="../assets/pill_lego_brick_mask.png" width="90%">
</p>
<p align="center"><em><b>Figure X.</b> Colour masking with Lego bricks. Left: camera image. Right: masked image for the task "yellow": every non-target brick is replaced by a flat patch of the surrounding background colour.</em></p>

Since real pills are white, HSV ranges can no longer tell them apart. Rather than writing a new masking method, I turned the real pills back into the coloured problem that already worked, at two different moments:

1. Once, before the rollout starts `main()` rewrites the task in `sys.argv`, replacing the pill name with a colour ("Put the amox pill in the circle compartment." becomes "Put the red pill in the circle compartment."). Each pill type has a fixed colour: amox to red, cal to blue, nast to yellow, ibu to green.
2. At every frame a YOLO11s detector, fine-tuned on my own pills, finds each pill and its type. `colour_boxes` paints each detected box with its type's colour, and `frame_mask` masks every colour except the target's, exactly as with the Lego bricks.

The per-frame step runs inside LeRobot through a monkey patch: I replaced `ThreadSafeRobot.get_observation` with my own function `get_masked_observation`, which takes the raw camera image, runs YOLO, paints and masks it, and returns the masked image to the policy. This changes what the policy sees without modifying the LeRobot library. My script `my_rollout.py` rewrites the task, applies the patch, then calls LeRobot's rollout.

YOLO alone was not stable enough: a pill's predicted type changed from frame to frame, so the target masked and unmasked itself during the episode. I therefore added three mechanisms:
- A tracking method YOLO runs in tracking mode, so each pill keeps the same ID across frames.
- Votes per pill `stable_class` keeps the last 15 predicted types for each ID and uses the most frequent one.
- Target lock a pill becomes the target only after 8 votes for the target type, and stays locked until it has been out of view for 30 frames. While the locked pill is visible, it is the only one painted with the target colour: any other pill predicted as the target is painted as a distractor and masked.

The lock has two gaps. Before a target is locked, and while the locked pill is hidden (for example by the gripper), every pill predicted as the target type keeps the target colour, so two pills can briefly look like the target. I designed this approach, and Claude helped implement the voting and lock logic.

The training set has 327 images labelled in Roboflow, with 4 classes (`amox`, `cal`, `ibu`, `nast`). They come from two sources: 99 frames from 17 recorded episodes, and 228 frames saved during my own rollouts, so that the detector also sees the arm and the lighting of real deployment. The validation set has 30 frames from 4 other recorded episodes. Frames from the same episode look alike, so the effective sample is closer to 4 scenes than to 30 images, and it contains no rollout frames.

I trained YOLO11n and YOLO11s for 150 epochs at 640 px. On the validation set, mAP50-95 is nearly tied (0.794 for both). YOLO11s has a higher recall (0.90 vs 0.86) but a lower precision (0.91 vs 0.95): it misses fewer pills but produces more false detections. Each type of error has a different effect on the masked image:
- A missed pill is not painted, so it stays visible to the policy as a white pill.
- A false detection with a non-target class is painted and then masked, which leaves a flat background-coloured patch where there was no pill, as near the gripper in Figure Y. It does not attract the policy, but it adds an artefact to the image.
- A false detection with the target class keeps the target colour, and the policy may go to it. The target lock reduces this risk, because a false target must collect 8 votes before it is locked, but it does not remove it: a false target that collects 8 votes first is locked like a real one.

<p align="center">
  <img src="../assets/yolo_medoc_masking.png" width="90%">
</p>
<p align="center"><em><b>Figure Y.</b> Real pills. Left: camera image. Middle: YOLO boxes painted with each pill type's colour. Right: masked image passed to the policy for the task "cal" (target painted blue).</em></p>

**Limitation**. I have not yet run controlled trials comparing task success with and without masking, so this section shows how the perception module works, not how much it improves sorting. The next step is to run the same layouts N times with masking and N times without, and to report the success rate of each.

### Prescription

The last part of the project is a web interface that I use on my phone. I photograph a prescription, and the PC reads it with OCR (Tesseract) and a parser that turns it into a table: for each pill, how many to take in the morning, at noon, in the evening and at night. Each time of day corresponds to one compartment of the organizer, marked with a symbol (morning to circle, noon to cross, evening to triangle, night to square), so the robot receives commands such as "Put the amox pill in the circle compartment."

Because the robot handles medication, the parser refuses to guess. It gives an error instead of a task in these cases: a drug name it does not recognise followed by a dosage, a known drug with no dosage, a half tablet, a frequency without times ("3x/day"), more than 4 pills of one drug at one time of day, or more pills than fit in one compartment. These rules are covered by 27 tests. The table is then shown on the phone, where I can check and correct it, and the robot only starts when I press Confirm. It then runs one rollout per pill, filling one day of the organizer. I designed the structure and the safety rules of this part, and Claude wrote the code, since it was outside the main scope of the project.

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

8. Emukpere, D., Deffayet, R., & Renders, J.-M. (2026). *Robust Skills, Brittle Grounding: Diagnosing Restricted Generalization in Vision-Language Action Policies via Multi-Object Picking*. arXiv:2602.24143. https://arxiv.org/abs/2602.24143
