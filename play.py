"""
Real-time inference loop for Subway Surfers CNN agent — Phase 2: Frame Stacking.

Changes from Phase 1:
- Maintains a rolling FIFO deque of the last STACK_SIZE preprocessed frames.
- Each inference call stacks those frames into [STACK_SIZE, 96, 96] before
  feeding them to the model.
- Model is loaded with in_channels=STACK_SIZE.
"""

import time
import collections
import torch
import torch.nn.functional as F
import pyautogui
import mss
import keyboard
from PIL import Image
from torchvision import transforms

from train_model import SubwayCNN, frame_transform, STACK_SIZE
from config import screen_width, screen_height

# ──────────────────────────────────────────────────────────────────────────────
# ACTION MAP
# ──────────────────────────────────────────────────────────────────────────────
ACTIONS = {
    0: "roll",
    1: "left",
    2: "noop",
    3: "right",
    4: "jump",
}

# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────────────────────────
model = SubwayCNN(in_channels=STACK_SIZE, num_actions=5)
checkpoint = torch.load("models/subway_cnn.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state"])
model.eval()
print(f"Model loaded | stack_size={STACK_SIZE} | val_acc={checkpoint.get('val_acc', 'N/A'):.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# FRAME BUFFER  (FIFO deque of STACK_SIZE tensors [1, 96, 96])
# ──────────────────────────────────────────────────────────────────────────────
frame_buffer = collections.deque(maxlen=STACK_SIZE)


def get_screen():
    """Capture full screen, preprocess, and return a [1, 96, 96] tensor."""
    with mss.mss() as sct:
        monitor = {"left": 0, "top": 0, "width": screen_width, "height": screen_height}
        screenshot = sct.grab(monitor)

    img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
    tensor = frame_transform(img)          # → [1, 96, 96]

    if not isinstance(tensor, torch.Tensor):
        tensor = transforms.ToTensor()(tensor)

    return tensor                          # [1, 96, 96]


def get_stacked_frame():
    """Return a [STACK_SIZE, 96, 96] tensor from the frame buffer."""
    new_frame = get_screen()               # [1, 96, 96]
    frame_buffer.append(new_frame)

    # Pad with the earliest frame if the buffer is not yet full
    while len(frame_buffer) < STACK_SIZE:
        frame_buffer.appendleft(frame_buffer[0])

    stacked = torch.cat(list(frame_buffer), dim=0)   # [STACK_SIZE, 96, 96]
    return stacked.unsqueeze(0)                       # [1, STACK_SIZE, 96, 96]


# ──────────────────────────────────────────────────────────────────────────────
# ACTION EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
def perform_action(action: int):
    key_map = {0: "down", 1: "left", 3: "right", 4: "up"}
    if action in key_map:
        pyautogui.press(key_map[action])


# ──────────────────────────────────────────────────────────────────────────────
# MAIN INFERENCE LOOP
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Inference loop running...")
    print("Press 'P' to pause/resume, 'Q' to quit.")

    paused       = False
    noop_printed = False

    while True:
        # ── hotkeys ──────────────────────────────────────────────────────────
        if keyboard.is_pressed("q"):
            print("Quitting...")
            break

        if keyboard.is_pressed("p"):
            paused = not paused
            print("Paused" if paused else "Resumed")
            time.sleep(0.3)   # debounce

        if paused:
            time.sleep(0.1)
            continue

        # ── capture + stack ──────────────────────────────────────────────────
        frame = get_stacked_frame()        # [1, STACK_SIZE, 96, 96]

        # ── inference ────────────────────────────────────────────────────────
        with torch.no_grad():
            output = model(frame)
            probs  = F.softmax(output, dim=1)[0]

        action     = torch.argmax(probs).item()
        confidence = probs[action].item()

        # ── confidence gate (90% threshold) ──────────────────────────────────
        if confidence < 0.9:
            action = 2   # noop
        action_name = ACTIONS[action]

        # ── logging ──────────────────────────────────────────────────────────
        if action_name == "noop":
            if not noop_printed:
                print(f"{action_name} | confidence={confidence:.3f}")
                noop_printed = True
        else:
            noop_printed = False
            print(f"{action_name} | confidence={confidence:.3f}")

        perform_action(action)
        time.sleep(0.02)
