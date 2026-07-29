# external/ -- third-party validation tools (not tracked in git)

This folder holds third-party code used only to independently validate
`uav_sim/dynamics.py` (see `uav_sim/validate_against_pybullet.py`). It is
NOT part of the project deliverable and is excluded from git via
`.gitignore` -- reproduce it with the steps below if this folder is missing.

## gym-pybullet-drones

```
git clone https://github.com/utiasDSL/gym-pybullet-drones.git external/gym-pybullet-drones
conda run -n uav_project pip install -e external/gym-pybullet-drones
conda run -n uav_project pip install -q "setuptools<81"   # pkg_resources compatibility
```

Then apply one manual correction: `gym_pybullet_drones/assets/cf2x.urdf`
ships with `mass value="0.027"` -- change this to `0.029` to match the
official Bitcraze Crazyflie 2.1 datasheet takeoff weight used in
`dynamics.py`'s `CrazyflieParams`, so the cross-validation compares the
same physical drone rather than two slightly different masses. (Inertia
values are already identical -- `dynamics.py`'s Ixx/Iyy/Izz were originally
sourced from this same cf2x.urdf.)

Verify with:
```
conda run -n uav_project python3 uav_sim/validate_against_pybullet.py
```
