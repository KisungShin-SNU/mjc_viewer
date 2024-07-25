# inspired by https://github.com/rohanpsingh/mujoco-python-viewer/blob/main/examples/sample.py

# load library
import h5py
import view_mjc

# set params
traj_fol_path = './traj/pilot_ft/'
traj_idx = 5
xml_scene_path = './mujoco_menagerie/universal_robots_ur5e/scene.xml'
# https://github.com/robot-descriptions/robot_descriptions.py
# from robot_descriptions import ur5e_mj_description
# xml_scene_path = ur5e_mj_description.MJCF_PATH

with h5py.File(traj_fol_path + f'traj_{traj_idx}.h5', 'r') as f:
    qpos_list = list(f[f'dict_str_traj_{traj_idx}']['dict_str_obs']['dict_str_state'])
    qvel_list = list(f[f'dict_str_traj_{traj_idx}']['dict_str_actions'])

# remove gripper's action
for idx, qpos in enumerate(qpos_list):
    qpos_list[idx] = qpos[:-1]

# run sim
view_mjc.render(xml_scene_path=xml_scene_path, qpos_list=qpos_list)