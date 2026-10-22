import mujoco
import numpy as np

# Measured from the MuJoCo origin (your blue Z), in metres.
AREA_X, AREA_Y = (0.18, 0.386), (-0.127, 0.076)     # inner edge of the tape
BOX_X, BOX_Y = (0.332, 0.377), (-0.081, 0.027)       # pill organizer footprint
PILLS = ["pill_red", "pill_blue", "pill_yellow", "pill_green"]
PILL_Z = 0.0052                                       # pill half-height + 0.2 mm


def random_pill_positions(model, data, rng, margin=0.02, min_dist=0.04, max_tries=1000):
    """Place every pill at a random spot in the taped area, flat on the table,
    with a random rotation, away from the pill box and from each other.

    margin   : keeps each pill fully inside the area and away from the box (m)
    min_dist : minimum distance between two pill centres, so they never overlap (m)
    Returns {pill_name: (x, y, yaw_deg)} so you can log the layout of each rollout.
    """
    placed, layout = [], {}
    for name in PILLS:
        for _ in range(max_tries):
            x = rng.uniform(AREA_X[0] + margin, AREA_X[1] - margin)
            y = rng.uniform(AREA_Y[0] + margin, AREA_Y[1] - margin)
            near_box = (BOX_X[0] - margin < x < BOX_X[1] + margin
                        and BOX_Y[0] - margin < y < BOX_Y[1] + margin)
            too_close = any(np.hypot(x - px, y - py) < min_dist for px, py in placed)
            if not near_box and not too_close:
                break
        else:
            raise RuntimeError(f"Could not place {name}: lower min_dist or margin")
        placed.append((x, y))

        yaw = rng.uniform(-np.pi, np.pi)
        joint = model.body(name).jntadr[0]
        q, v = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        data.qpos[q:q + 3] = [x, y, PILL_Z]
        data.qpos[q + 3:q + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]   # rotation about z
        data.qvel[v:v + 6] = 0
        layout[name] = (x, y, np.degrees(yaw))

    mujoco.mj_forward(model, data)
    return layout