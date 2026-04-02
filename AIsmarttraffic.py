import cv2
import numpy as np
from ultralytics import YOLO
import time

# -----------------------------
# Load YOLO Model
# -----------------------------
model = YOLO("yolov8n.pt")  # lightweight model

# -----------------------------
# Reinforcement Learning Setup
# -----------------------------
Q = np.zeros((50, 3))  # states (vehicle count ranges), actions (short, medium, long)

alpha = 0.1
gamma = 0.9
epsilon = 0.1

def get_state(vehicle_count):
    return min(vehicle_count // 2, 49)

def choose_action(state):
    if np.random.rand() < epsilon:
        return np.random.randint(3)
    return np.argmax(Q[state])

def update_q(state, action, reward, next_state):
    Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

# -----------------------------
# Signal Timing Logic
# -----------------------------
def get_signal_time(action):
    if action == 0:
        return 20
    elif action == 1:
        return 40
    else:
        return 60

# -----------------------------
# Vehicle Detection Function
# -----------------------------
def detect_vehicles(frame):
    results = model(frame)
    vehicle_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            # Classes: car, bike, bus, truck
            if cls in [2, 3, 5, 7]:
                vehicle_count += 1

    return vehicle_count

# -----------------------------
# Main Execution
# -----------------------------
cap = cv2.VideoCapture("traffic.mp4")  # replace with 0 for webcam

prev_state = 0
prev_action = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    vehicle_count = detect_vehicles(frame)
    state = get_state(vehicle_count)

    action = choose_action(state)
    signal_time = get_signal_time(action)

    # Reward (less waiting = better)
    reward = -vehicle_count

    update_q(prev_state, prev_action, reward, state)

    prev_state = state
    prev_action = action

    # Display info
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(frame, f"Signal Time: {signal_time}s", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    if signal_time == 20:
        signal_text = "GREEN (SHORT)"
    elif signal_time == 40:
        signal_text = "GREEN (MEDIUM)"
    else:
        signal_text = "GREEN (LONG)"

    cv2.putText(frame, signal_text, (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Smart Traffic System", frame)

    # Simulate delay (signal duration)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.1)

cap.release()
cv2.destroyAllWindows()