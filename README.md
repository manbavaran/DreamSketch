# DreamSketch

An interactive AI gesture-based visual art system that allows users to draw in the air using hand gestures, trigger particle effects like sakura, roses, meteors, and hearts — all powered by MediaPipe and OpenCV.

## Features

- ✍️ **OK + Index Up Gesture**: Activate "Draw" mode with a glowing trail.
- 🌸 **One-Hand Heart Gesture**: Emits sakura (right hand) or rose (left hand) particles.
- 💫 **Palm Sweep Gesture**: Triggers meteor rain in the direction of the hand sweep.
- 💖 **Two-Hand Heart Gesture**: Emits heart particles.
- 🎮 Real-time gesture detection using MediaPipe.
- 🌟 Particle effects rendered with OpenCV for real-time interaction.

## Installation
pip install -r requirements.txt


# Run the App
python main.py

# Dependencies
See requirements.txt

# Project Structure
├── main.py                  # Main loop handling gestures and rendering
├── gesture.py              # Custom gesture definitions and logic
├── particle.py             # Particle system (meteor, sakura, rose, heart)
├── trajectory_glow.py      # Glowing trail for draw mode
├── requirements.txt

# License
This project is licensed under the MIT License. See LICENSE for details.

# Acknowledgements
MediaPipe Hands

OpenCV