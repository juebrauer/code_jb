# Overview

## `01_helloworld_pybullet.py`
A simple world with a robot falling from z=1.5m to the ground

Call it with:
`python 01_helloworld_pybullet.py`



## `02_falling_objects_pybullet.py`
n objects will fall down from the sky

Example usage:
`python 02_falling_objects_pybullt.py --n 100 --seconds 5 --seed 42`



## `03_franka_robot_table_with_objects.py`
A Franka robot arm that picks and places a single object

Example usage:
`python 03_franka_robot_table_with_objects.py`



## `04_franka_robot_pick_red_with_recording.py`
The same robot arm that picks and places a red object per episode.
Camera views and actions are being recorded.

Example usage:
`python 04_franka_robot_pick_red_with_recording.py --episodes 100 --seed 42 --save-every 10`

The recorded sequences will be stored in a folder with name `data`.



## `05_train_and_test_cnn.py`




# Error: Failed to retrieve a framebuffer config

If you get an

    Failed to retrieve a framebuffer config

error, try to set the following environment variables:

    export __NV_PRIME_RENDER_OFFLOAD=1
    export __GLX_VENDOR_LIBRARY_NAME=nvidia

E.g., by adding them to your ~/.bashrc file

Or call

    source set_env_variables.sh

In my case this error vanished. However, PyBullet hung up from time to time ...

For this, continue reading!


# Error: PyBullet hangs up from time to time

For me the following command was a good solution:

    sudo prime-select nvidia

With this command, all displays/monitors are rendered with the dGPU.

Before, the internal GPU (iGPU) was rendering the 1st monitor and the NVIDIA GPU (dGPU) was rendering the 2nd monitor and this
caused problems to PyBullet. With this command I also did not need to set the environment variables as before since everything
is rendered by the NVIDIA GPU.


# __NV_PRIME_RENDER_OFFLOAD

The environment variable __NV_PRIME_RENDER_OFFLOAD=1 is a signal to the NVIDIA driver that tells it:

"I want to use Prime Render Offload mode for this application."

What "Offload Mode" Means: "Offload" refers to delegating the rendering work from the integrated GPU to the dedicated NVIDIA GPU, while the integrated GPU continues to handle display output.

Without offload mode set to 0:
- Application uses whatever GPU is currently active for display
- Usually the integrated GPU (Intel/AMD)
- Lower performance but better power efficiency

With offload mode set to 1:
- Application's 3D rendering is "offloaded" to the NVIDIA GPU
- NVIDIA GPU does all the heavy computational work
- Rendered frames are then copied back to integrated GPU for display
- Much higher performance for graphics-intensive tasks

# __GLX_VENDOR_LIBRARY_NAME

The environment variable __GLX_VENDOR_LIBRARY_NAME=nvidia tells the system which OpenGL implementation to use for rendering.

What GLX Is:

GLX (OpenGL Extension to the X Window System) is the interface that connects OpenGL applications to the X11 windowing system on Linux. It's what allows 3D applications to actually display their rendered content.

The Problem GLX Solves:
On a hybrid graphics system, you might have multiple OpenGL implementations installed:
- Mesa (open-source drivers for integrated Intel/AMD GPUs)
- NVIDIA proprietary drivers (for NVIDIA GPUs)

The system needs to know which one to use. 

What nvidia Means:
By setting __GLX_VENDOR_LIBRARY_NAME=nvidia, you're explicitly telling the system:
"For this application, use the NVIDIA OpenGL implementation, not the Mesa one."