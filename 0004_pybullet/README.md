# Overview

01_helloworld_pybullet: A simple world with a robot falling from z=1.5m to the ground
02_falling_objects_pybullet: n objects will fall down from the sky


# Errors?

If you get an

    Failed to retrieve a framebuffer config

error, try to set the following environment variables:

    export __NV_PRIME_RENDER_OFFLOAD=1
    export __GLX_VENDOR_LIBRARY_NAME=nvidia

E.g., by adding them to your ~/.bashrc file

Or call

    source set_env_variables.sh


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