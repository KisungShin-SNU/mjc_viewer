# inspired by https://colab.research.google.com/github/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb

import numpy as np
import tensorflow_datasets as tfds
#from PIL import Image
#from IPython import display
import view_mjc

DATASETS = [
    'fractal20220817_data',
    'kuka',
    'bridge',
    'taco_play',
    'jaco_play',
    'berkeley_cable_routing',
    'roboturk',
    'nyu_door_opening_surprising_effectiveness',
    'viola',
    'berkeley_autolab_ur5',
    'toto',
    'language_table',
    'columbia_cairlab_pusht_real',
    'stanford_kuka_multimodal_dataset_converted_externally_to_rlds',
    'nyu_rot_dataset_converted_externally_to_rlds',
    'stanford_hydra_dataset_converted_externally_to_rlds',
    'austin_buds_dataset_converted_externally_to_rlds',
    'nyu_franka_play_dataset_converted_externally_to_rlds',
    'maniskill_dataset_converted_externally_to_rlds',
    'cmu_franka_exploration_dataset_converted_externally_to_rlds',
    'ucsd_kitchen_dataset_converted_externally_to_rlds',
    'ucsd_pick_and_place_dataset_converted_externally_to_rlds',
    'austin_sailor_dataset_converted_externally_to_rlds',
    'austin_sirius_dataset_converted_externally_to_rlds',
    'bc_z',
    'usc_cloth_sim_converted_externally_to_rlds',
    'utokyo_pr2_opening_fridge_converted_externally_to_rlds',
    'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds',
    'utokyo_saytap_converted_externally_to_rlds',
    'utokyo_xarm_pick_and_place_converted_externally_to_rlds',
    'utokyo_xarm_bimanual_converted_externally_to_rlds',
    'robo_net',
    'berkeley_mvp_converted_externally_to_rlds',
    'berkeley_rpt_converted_externally_to_rlds',
    'kaist_nonprehensile_converted_externally_to_rlds',
    'stanford_mask_vit_converted_externally_to_rlds',
    'tokyo_u_lsmo_converted_externally_to_rlds',
    'dlr_sara_pour_converted_externally_to_rlds',
    'dlr_sara_grid_clamp_converted_externally_to_rlds',
    'dlr_edan_shared_control_converted_externally_to_rlds',
    'asu_table_top_converted_externally_to_rlds',
    'stanford_robocook_converted_externally_to_rlds',
    'eth_agent_affordances',
    'imperialcollege_sawyer_wrist_cam',
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds',
    'uiuc_d3field',
    'utaustin_mutex',
    'berkeley_fanuc_manipulation',
    'cmu_play_fusion',
    'cmu_stretch',
    'berkeley_gnm_recon',
    'berkeley_gnm_cory_hall',
    'berkeley_gnm_sac_son'
]


def dataset2path(dataset_name):
  if dataset_name == 'robo_net':
    version = '1.0.0'
  elif dataset_name == 'language_table':
    version = '0.0.1'
  else:
    version = '0.1.0'
  return f'gs://gresearch/robotics/{dataset_name}/{version}'

# !pip install numpy==1.25.2

# set params
num_load_episode = 10
traj_idx = 0

# TOTO Benchmark
dataset = 'toto'
# display_key = 'image'
xml_scene_path = './mujoco_menagerie/franka_fr3/scene.xml'

# Saytap
# dataset = 'berkeley_mvp_converted_externally_to_rlds'
# display_key = 'image'
# xml_scene_path = './mujoco_menagerie/ufactory_xarm7/scene.xml'

# Berkeley MVP Data
# display_key = 'hand_image'
# dataset = 'berkeley_mvp_converted_externally_to_rlds'
# xml_scene_path = './mujoco_menagerie/ufactory_xarm7/scene.xml'

# Berkeley RPT Data
# display_key = 'hand_image'
# dataset = 'berkeley_rpt_converted_externally_to_rlds'
# xml_scene_path = './mujoco_menagerie/franka_fr3/scene.xml'

b = tfds.builder_from_directory(builder_dir=dataset2path(dataset))
ds = b.as_dataset(split = f'train[:{num_load_episode}]')
episode_list = [next(iter(ds)) for i in range(num_load_episode)]

# TOTO Benchmark
traj_list = [[step['observation']['state'] for step in episode['steps']] for episode in episode_list]

# Saytap
# traj_list = [[step['action'] for step in episode['steps']] for episode in episode_list]

# Berkeley MVP Data
# traj_list = [[step['observation']['joint_pos'] for step in episode['steps']] for episode in episode_list]

# Berkeley RPT Data
# traj_list = [[step['observation']['joint_pos'] for step in episode['steps']] for episode in episode_list]

print(traj_list[traj_idx])
print(len(traj_list))
print(len(traj_list[traj_idx]))

# run sim
view_mjc.render(xml_scene_path=xml_scene_path, qpos_list=traj_list[traj_idx])