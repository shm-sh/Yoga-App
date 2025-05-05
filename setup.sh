#!/bin/bash

# Install system dependencies
apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk2.0-0 \
    libgtk-3-0 \
    libopenblas-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev