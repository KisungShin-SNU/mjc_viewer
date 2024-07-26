Ubuntu 20.04 LTS

git clone this_repository
cd this_repository

conda create -n "mjc_viewer" python=3.9
conda activate mjc_viewer

git clone https://github.com/google-deepmind/mujoco_menagerie.git

pip install -r requirements.txt

mkdir traj/
fill traj

modifying params & run
python traj_sim.py
python rlds_viewer.py

TODO
- add gripper
- fix rlds viewer