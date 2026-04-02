# AI Smart Traffic Management 🚦

This project uses YOLOv8 and Reinforcement Learning to dynamically control traffic signals.

## Features
- Real-time vehicle detection
- Adaptive signal timing
- AI-based decision making

## Tech Stack
Python, OpenCV, YOLOv8, RL

## Run
python main.py


Smart-Traffic-System/
│
├── data/                   # Sample traffic videos/images
├── models/                 # Trained YOLO & RL models
├── src/
│   ├── detection.py        # Vehicle detection
│   ├── rl_agent.py         # Reinforcement learning logic
│   ├── traffic_control.py  # Signal control logic
│
├── app.py                  # Flask dashboard
├── requirements.txt
├── README.md
└── demo.mp4
