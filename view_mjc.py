# inspired by https://github.com/rohanpsingh/mujoco-python-viewer/blob/main/examples/sample.py

# load library
import mujoco
import mujoco_viewer


def render(xml_scene_path, qpos_list):

    # create model and data
    model = mujoco.MjModel.from_xml_path(xml_scene_path)
    data = mujoco.MjData(model)

    # create the viewer object
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # simulate and render
    for i in range(len(qpos_list)):
        data.qpos = qpos_list[i]
        print(data.qpos)

        mujoco.mj_step(model, data)
        viewer.render()
        if not viewer.is_alive:
            break

    # close viewer
    viewer.close()