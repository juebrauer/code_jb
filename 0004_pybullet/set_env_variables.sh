#!/bin/bash

# set NVIDIA Prime Render Offload mode
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# check whether env variables have been set correctly
echo "NVIDIA Prime Render:"
echo "__NV_PRIME_RENDER_OFFLOAD=$__NV_PRIME_RENDER_OFFLOAD"
echo "__GLX_VENDOR_LIBRARY_NAME=$__GLX_VENDOR_LIBRARY_NAME"
