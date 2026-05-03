import cv2
import mediapipe as mp
import numpy as np
import math
import sys
import threading
import requests
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, Tk

# ── Stdout encoding fix (Handles Windows-specific encoding issues) ─────────────
sys.stdout.reconfigure(encoding='utf-8')

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ── Camera setup ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
# Set your desired resolution
WIDTH, HEIGHT = 1920, 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Transparent drawing layer overlaid on the camera feed
canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

# ── Constants ──────────────────────────────────────────────────────────────────
FINGER_NAMES = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}
DRAW_COLOR     = (0, 255, 0)
DRAW_THICKNESS = 4
ERASER_SIZE    = 40

# Drawing mode colors
DRAW_COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 255, 255),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 165, 255),    # Orange
    (255, 255, 0),    # Cyan
    (0, 150, 255),    # Light Orange
    (255, 100, 0),    # Dark Blue
]

# Settings button region (top-right corner)
BTN_X1, BTN_Y1 = WIDTH - 150, 10
BTN_X2, BTN_Y2 = WIDTH - 10, 50

# ── Flask Color Picker Server Setup ────────────────────────────────────────────
try:
    # Try to import and start Flask server
    from color_picker_server import app as flask_app, run_server
    
    # Start Flask server in background thread
    flask_thread = threading.Thread(target=run_server, kwargs={'port': 5000}, daemon=True)
    flask_thread.start()
    print("[✓] Flask Color Picker server started on http://127.0.0.1:5000")
    print("[ℹ] Open browser to http://127.0.0.1:5000 for color selection")
    FLASK_AVAILABLE = True
except Exception as e:
    print(f"[!] Flask server not available: {e}")
    print("[ℹ] Color picker will use default palette")
    FLASK_AVAILABLE = False

import time
time.sleep(1)  # Give Flask time to start

# ── Mutable app state ──────────────────────────────────────────────────────────
state = {
    "main_mode":       "DRAWING",  # DRAWING | INTERACTIVE
    "selected_mode":   None,       # Mode pending selection in menu (None if not selected)
    "sub_mode":        "FREE",     # FREE | DRAW | ERASE | COLOR_PICKER (for DRAWING mode)
    "prev_point":      None,       # last drawn point for continuous lines
    "save_flash":      0,          # frame counter for the save confirmation flash
    "screenshots_dir": str(Path.home() / "Desktop"),
    "btn_clicked":     False,
    "mouse_pos":       (0, 0),
    "show_ui":         False,      # Feature to toggle settings overlay
    "fullscreen_toggle": True,     # Toggle fullscreen mode in settings
    "menu_open":       False,      # Main mode selection menu
    "ok_gesture_hold":  0,         # Frame counter for OK gesture hold
    "ok_gesture_pressed": False,   # Flag to prevent multiple triggers from same hold
    "gesture_1_hold":   0,         # Frame counter for gesture 1 hold
    "gesture_2_hold":   0,         # Frame counter for gesture 2 hold
    # Drawing mode color picker state
    "draw_color_idx": 0,           # Current draw color index
    "color_picker_hold": 0,        # Frame counter for 4-finger gesture hold
    "color_picker_pressed": False, # Flag to prevent multiple triggers
    "flask_color_bgr": (0, 255, 0),   # Flask web picker color (BGR)
    "flask_color_hex": "#00FF00",      # Flask web picker color (HEX)
    "last_flask_check": time.time(),  # Last Flask API check
}
Path(state["screenshots_dir"]).mkdir(parents=True, exist_ok=True)

# ── Helper functions ───────────────────────────────────────────────────────────

def get_pos(landmark, w, h):
    """Convert a normalized landmark to pixel coordinates."""
    return int(landmark.x * w), int(landmark.y * h)

def only_index_up(landmarks):
    """Return True if only the index finger is extended."""
    return all([
        landmarks[8].y  < landmarks[6].y,
        landmarks[12].y > landmarks[10].y,
        landmarks[16].y > landmarks[14].y,
        landmarks[20].y > landmarks[18].y,
    ])

def is_gesture_1(landmarks):
    """Gesture 1: Only index finger extended (for left hand mode selection)."""
    return all([
        landmarks[8].y  < landmarks[6].y,
        landmarks[12].y > landmarks[10].y,
        landmarks[16].y > landmarks[14].y,
        landmarks[20].y > landmarks[18].y,
    ])

def is_gesture_2(landmarks):
    """Gesture 2: Index and middle fingers extended (for left hand mode selection)."""
    return all([
        landmarks[8].y  < landmarks[6].y,
        landmarks[12].y < landmarks[10].y,
        landmarks[16].y > landmarks[14].y,
        landmarks[20].y > landmarks[18].y,
    ])

def is_gesture_4(landmarks):
    """Gesture 4: 4 fingers extended, thumb tucked in (For Color Palette)."""
    # Check if thumb is tucked in based on hand orientation
    thumb_tucked = landmarks[4].x > landmarks[3].x if landmarks[17].x > landmarks[5].x else landmarks[4].x < landmarks[3].x
    return all([
        thumb_tucked,
        landmarks[8].y  < landmarks[6].y,   # Index up
        landmarks[12].y < landmarks[10].y,  # Middle up
        landmarks[16].y < landmarks[14].y,  # Ring up
        landmarks[20].y < landmarks[18].y,  # Pinky up
    ])

def is_ok_gesture(landmarks, w, h):
    """Return True if OK gesture is detected (thumb and index close together)."""
    thumb_pos = get_pos(landmarks[4], w, h)
    index_pos = get_pos(landmarks[8], w, h)
    distance = math.sqrt((thumb_pos[0] - index_pos[0])**2 + (thumb_pos[1] - index_pos[1])**2)
    # Just need thumb and index close together
    return distance < 70

def is_open_hand(landmarks):
    """Return True if all four fingers are extended (open palm)."""
    return all([
        landmarks[8].y  < landmarks[6].y,
        landmarks[12].y < landmarks[10].y,
        landmarks[16].y < landmarks[14].y,
        landmarks[20].y < landmarks[18].y,
    ])

def hand_center(landmarks, w, h):
    """Return the center pixel of the palm."""
    points = [0, 5, 9, 13, 17]
    cx = int(sum(landmarks[i].x for i in points) / len(points) * w)
    cy = int(sum(landmarks[i].y for i in points) / len(points) * h)
    return cx, cy

def hand_size(landmarks, w, h):
    """Estimate palm width in pixels (used for eraser radius)."""
    x1, y1 = int(landmarks[5].x * w),  int(landmarks[5].y * h)
    x2, y2 = int(landmarks[17].x * w), int(landmarks[17].y * h)
    return int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

def draw_hand_labels(frame, hand_landmarks, label, w, h):
    """Render hand label and fingertip names on the frame with enhanced styling."""
    color = (255, 150, 0) if label == "LEFT" else (0, 150, 255)
    shadow_color = (50, 50, 50)
    wx = int(hand_landmarks.landmark[0].x * w)
    wy = int(hand_landmarks.landmark[0].y * h)
    # Shadow effect for hand label
    cv2.putText(frame, label, (wx - 18, wy + 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, shadow_color, 3)
    cv2.putText(frame, label, (wx - 20, wy + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
    for idx, name in FINGER_NAMES.items():
        x = int(hand_landmarks.landmark[idx].x * w)
        y = int(hand_landmarks.landmark[idx].y * h)
        # Shadow effect for finger names
        cv2.putText(frame, name, (x + 6, y - 3),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, shadow_color, 2)
        cv2.putText(frame, name, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 1)

def draw_settings_button(frame, hover=False):
    """Draw the settings button in the top-right corner with enhanced styling."""
    color = (150, 150, 150) if hover else (100, 100, 100)
    bg_color = (45, 45, 45) if hover else (35, 35, 35)
    thickness = 3 if hover else 2
    # Button background with gradient effect
    cv2.rectangle(frame, (BTN_X1, BTN_Y1), (BTN_X2, BTN_Y2), bg_color, -1)
    # Button border with enhanced visibility
    cv2.rectangle(frame, (BTN_X1, BTN_Y1), (BTN_X2, BTN_Y2), color, thickness)
    # Shadow effect
    cv2.rectangle(frame, (BTN_X1 + 2, BTN_Y1 + 2), (BTN_X2 - 1, BTN_Y2 - 1), (20, 20, 20), 1)
    # Button text with shadow
    text_color = (255, 255, 255) if hover else (230, 230, 230)
    cv2.putText(frame, "[ SETTINGS ]", (BTN_X1 + 5, BTN_Y1 + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (50, 50, 50), 2)
    cv2.putText(frame, "[ SETTINGS ]", (BTN_X1 + 4, BTN_Y1 + 27),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, text_color, 1)

def draw_settings_ui(frame):
    """Semi-transparent Settings UI Panel with enhanced styling and animations."""
    mx, my = state["mouse_pos"]
    overlay = frame.copy()
    
    # Main Panel Background with border
    panel_x1, panel_y1 = WIDTH//4, HEIGHT//4
    panel_x2, panel_y2 = 3*WIDTH//4, 3*HEIGHT//4
    # Shadow effect
    cv2.rectangle(overlay, (panel_x1 + 3, panel_y1 + 3), (panel_x2 + 3, panel_y2 + 3), (10, 10, 10), -1)
    # Main panel
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (35, 35, 35), -1)
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (220, 220, 220), 3)
    
    # Top accent bar
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y1 + 5), (100, 200, 255), -1)
    
    # Title with shadow
    title_x = panel_x1 + 50
    title_y = panel_y1 + 50
    cv2.putText(overlay, "SETTINGS MENU", (title_x + 2, title_y + 2), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (30, 30, 30), 3)
    cv2.putText(overlay, "SETTINGS MENU", (title_x, title_y), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

    # Change Folder Button Logic with enhanced styling
    f_rect = (WIDTH//2 - 160, HEIGHT//2 - 50, WIDTH//2 + 160, HEIGHT//2 + 10)
    f_hover = f_rect[0] < mx < f_rect[2] and f_rect[1] < my < f_rect[3]
    f_color = (80, 200, 80) if f_hover else (60, 120, 60)
    f_text_color = (255, 255, 255) if f_hover else (220, 220, 220)
    f_thickness = 3 if f_hover else 2
    # Button shadow
    cv2.rectangle(overlay, (f_rect[0] + 2, f_rect[1] + 2), (f_rect[2] + 2, f_rect[3] + 2), (20, 20, 20), -1)
    cv2.rectangle(overlay, (f_rect[0], f_rect[1]), (f_rect[2], f_rect[3]), f_color, -1)
    cv2.rectangle(overlay, (f_rect[0], f_rect[1]), (f_rect[2], f_rect[3]), (255, 255, 255) if f_hover else (180, 180, 180), f_thickness)
    cv2.putText(overlay, "Change Save Folder", (f_rect[0] + 25, f_rect[1] + 38), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (30, 30, 30) if f_hover else (20, 20, 20), 2)
    cv2.putText(overlay, "Change Save Folder", (f_rect[0] + 24, f_rect[1] + 37), 
                cv2.FONT_HERSHEY_DUPLEX, 0.8, f_text_color, 1)

    # Back Button Logic with enhanced styling
    b_rect = (WIDTH//2 - 70, 3*HEIGHT//4 - 70, WIDTH//2 + 70, 3*HEIGHT//4 - 20)
    b_hover = b_rect[0] < mx < b_rect[2] and b_rect[1] < my < b_rect[3]
    b_color = (200, 80, 80) if b_hover else (120, 60, 120)
    b_text_color = (255, 255, 255) if b_hover else (220, 220, 220)
    b_thickness = 3 if b_hover else 2
    # Button shadow
    cv2.rectangle(overlay, (b_rect[0] + 2, b_rect[1] + 2), (b_rect[2] + 2, b_rect[3] + 2), (20, 20, 20), -1)
    cv2.rectangle(overlay, (b_rect[0], b_rect[1]), (b_rect[2], b_rect[3]), b_color, -1)
    cv2.rectangle(overlay, (b_rect[0], b_rect[1]), (b_rect[2], b_rect[3]), (255, 255, 255) if b_hover else (180, 180, 180), b_thickness)
    cv2.putText(overlay, "BACK", (b_rect[0] + 16, b_rect[1] + 38), 
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (30, 30, 30), 2)
    cv2.putText(overlay, "BACK", (b_rect[0] + 15, b_rect[1] + 37), 
                cv2.FONT_HERSHEY_DUPLEX, 0.85, b_text_color, 1)

    # Fullscreen Toggle Switch - Simple Rectangle Toggle
    toggle_label_x = WIDTH//2 - 250
    toggle_label_y = HEIGHT//2 + 60
    cv2.putText(overlay, "Fullscreen:", (toggle_label_x, toggle_label_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (220, 220, 220), 2)
    
    # Toggle switch dimensions - rectangular and compact
    switch_x1 = WIDTH//2 + 20
    switch_y1 = toggle_label_y - 30
    switch_x2 = switch_x1 + 180
    switch_y2 = switch_y1 + 60
    switch_mid = (switch_x1 + switch_x2) // 2
    
    # Check if toggle is hovered
    toggle_hover = (switch_x1 - 10 < mx < switch_x2 + 10 and 
                    switch_y1 - 10 < my < switch_y2 + 10)
    
    # LEFT SIDE - OFF (Red)
    off_color = (120, 40, 40) if toggle_hover and not state["fullscreen_toggle"] else (80, 30, 30)
    cv2.rectangle(overlay, (switch_x1, switch_y1), (switch_mid, switch_y2), off_color, -1)
    cv2.rectangle(overlay, (switch_x1, switch_y1), (switch_mid, switch_y2), (200, 100, 100), 2)
    cv2.putText(overlay, "OFF", (switch_x1 + 10, switch_y1 + 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (180, 100, 100), 2)
    
    # RIGHT SIDE - ON (Green)
    on_color = (40, 120, 40) if toggle_hover and state["fullscreen_toggle"] else (30, 80, 30)
    cv2.rectangle(overlay, (switch_mid, switch_y1), (switch_x2, switch_y2), on_color, -1)
    cv2.rectangle(overlay, (switch_mid, switch_y1), (switch_x2, switch_y2), (100, 200, 100), 2)
    cv2.putText(overlay, "ON", (switch_x2 - 45, switch_y1 + 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (100, 200, 100), 2)
    
    # Sliding button indicator
    if state["fullscreen_toggle"]:
        slider_x1 = switch_mid + 3
        slider_x2 = switch_x2 - 3
        slider_color = (100, 255, 100)
    else:
        slider_x1 = switch_x1 + 3
        slider_x2 = switch_mid - 3
        slider_color = (255, 100, 100)
    
    slider_y1 = switch_y1 + 6
    slider_y2 = switch_y2 - 6
    
    # Draw slider bar
    cv2.rectangle(overlay, (slider_x1, slider_y1), (slider_x2, slider_y2), slider_color, -1)
    cv2.rectangle(overlay, (slider_x1, slider_y1), (slider_x2, slider_y2), (255, 255, 255), 2)

    if state["btn_clicked"]:
        if f_hover:
            chosen = open_folder_dialog()
            if chosen: state["screenshots_dir"] = chosen
        elif b_hover:
            state["show_ui"] = False
        elif toggle_hover:
            state["fullscreen_toggle"] = not state["fullscreen_toggle"]
            if state["fullscreen_toggle"]:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        state["btn_clicked"] = False

    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

def draw_main_menu(frame):
    """Draw the main mode selection menu."""
    overlay = frame.copy()
    
    # Menu dimensions
    menu_width = 500
    menu_height = 400
    menu_x1 = WIDTH // 2 - menu_width // 2
    menu_y1 = HEIGHT // 2 - menu_height // 2
    menu_x2 = menu_x1 + menu_width
    menu_y2 = menu_y1 + menu_height
    
    # Shadow effect
    cv2.rectangle(overlay, (menu_x1 + 3, menu_y1 + 3), (menu_x2 + 3, menu_y2 + 3), (10, 10, 10), -1)
    
    # Main menu background
    cv2.rectangle(overlay, (menu_x1, menu_y1), (menu_x2, menu_y2), (30, 30, 30), -1)
    cv2.rectangle(overlay, (menu_x1, menu_y1), (menu_x2, menu_y2), (100, 200, 255), 4)
    
    # Title bar
    cv2.rectangle(overlay, (menu_x1, menu_y1), (menu_x2, menu_y1 + 70), (50, 100, 150), -1)
    
    # Title
    cv2.putText(overlay, "MAIN MODE MENU", (menu_x1 + 80, menu_y1 + 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
    
    # Mode buttons
    modes = ["DRAWING", "INTERACTIVE"]
    colors = [(0, 255, 100), (255, 150, 0)]
    
    button_height = 80
    button_y_start = menu_y1 + 100
    
    for idx, (mode, color) in enumerate(zip(modes, colors)):
        button_y1 = button_y_start + idx * (button_height + 40)
        button_y2 = button_y1 + button_height
        
        # Highlight selected mode (pending)
        if state["selected_mode"] == mode:
            highlight_color = tuple(min(c + 50, 255) for c in color)
            cv2.rectangle(overlay, (menu_x1 + 20, button_y1), (menu_x2 - 20, button_y2), highlight_color, -1)
            cv2.rectangle(overlay, (menu_x1 + 20, button_y1), (menu_x2 - 20, button_y2), (255, 255, 255), 4)
        else:
            cv2.rectangle(overlay, (menu_x1 + 20, button_y1), (menu_x2 - 20, button_y2), (50, 50, 50), -1)
            cv2.rectangle(overlay, (menu_x1 + 20, button_y1), (menu_x2 - 20, button_y2), color, 3)
        
        # Button label (gesture number)
        gesture_num = idx + 1
        cv2.putText(overlay, f"Gesture {gesture_num}", (menu_x1 + 40, button_y1 + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 150, 150), 1)
        
        # Button text
        text_color = (255, 255, 255) if state["selected_mode"] == mode else color
        cv2.putText(overlay, mode, (menu_x1 + 110, button_y1 + 65),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, text_color, 2)
    
    # Instructions
    cv2.putText(overlay, "Left hand: 1 finger = DRAWING | 2 fingers = INTERACTIVE",
                (menu_x1 + 20, menu_y2 - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (100, 200, 150), 1)
    
    cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)

def open_folder_dialog():
    """Open a folder selection dialog and return the chosen path."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    chosen = filedialog.askdirectory(
        title="Select folder to save screenshots",
        initialdir=state["screenshots_dir"]
    )
    root.destroy()
    return chosen if chosen else None

def imwrite_unicode(path, img):
    """cv2.imwrite wrapper that handles Unicode paths on Windows."""
    ext = Path(path).suffix.lower()
    success, buf = cv2.imencode(ext, img)
    if not success:
        print(f"[ERROR] Encoding failed for: {path}")
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True

def save_screenshot(frame, canvas, canvas_only=False):
    """Save the current view to state['screenshots_dir']."""
    save_dir = state["screenshots_dir"]
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix    = "drawing" if canvas_only else "snapshot"
    filename  = str(Path(save_dir) / f"{prefix}_{timestamp}.png")
    target = canvas if canvas_only else frame
    result = imwrite_unicode(filename, target)
    if result:
        print(f"[OK] Saved to: {filename}")
    return filename

# ── Interactive mode helper functions ──────────────────────────────────────────

def draw_grid_color_palette(frame, center):
    """Draw a 4x4 square color palette in the center of the screen."""
    grid_size = 4
    cell_size = 65
    padding = 12
    
    # Calculate start positions for centering
    total_dim = (grid_size * cell_size) + ((grid_size - 1) * padding)
    start_x = center[0] - total_dim // 2
    start_y = center[1] - total_dim // 2
    
    # Draw background panel
    cv2.rectangle(frame, (start_x - 25, start_y - 70), 
                  (start_x + total_dim + 25, start_y + total_dim + 25), (30, 30, 30), -1)
    cv2.rectangle(frame, (start_x - 25, start_y - 70), 
                  (start_x + total_dim + 25, start_y + total_dim + 25), (150, 150, 150), 2)
    
    cv2.putText(frame, "SELECT COLOR (Use Left Index)", (start_x, start_y - 30), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    for i in range(len(DRAW_COLORS)):
        row = i // grid_size
        col = i % grid_size
        
        x1 = start_x + col * (cell_size + padding)
        y1 = start_y + row * (cell_size + padding)
        x2 = x1 + cell_size
        y2 = y1 + cell_size
        
        # Highlight selected color border
        if i == state["draw_color_idx"]:
            cv2.rectangle(frame, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (255, 255, 255), 3)
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), DRAW_COLORS[i], -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

# ── Mouse callback ─────────────────────────────────────────────────────────────
def on_mouse(event, x, y, flags, param):
    state["mouse_pos"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        state["btn_clicked"] = True

# ── Window setup (Fullscreen) ──────────────────────────────────────────────────
WINDOW_NAME = "Hand Drawing"
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

# ── Main loop ──────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    h, w    = frame.shape[:2]
    
    # Fetch color from Flask web picker (every 200ms) ──
    if FLASK_AVAILABLE and state["sub_mode"] == "COLOR_PICKER":
        current_time = time.time()
        if (current_time - state["last_flask_check"]) > 0.2:
            try:
                response = requests.get('http://127.0.0.1:5000/api/color', timeout=0.5)
                if response.status_code == 200:
                    data = response.json()
                    bgr_tuple = tuple(data.get('bgr', (0, 255, 0)))
                    hex_color = data.get('hex', '#00FF00')
                    state["current_draw_color"] = bgr_tuple
                    state["flask_color_hex"] = hex_color
            except:
                pass
            state["last_flask_check"] = current_time

    if not state["show_ui"]:
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left_landmarks  = None
        right_landmarks = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label_raw = results.multi_handedness[i].classification[0].label
                label     = "LEFT" if label_raw == "Left" else "RIGHT"

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                draw_hand_labels(frame, hand_landmarks, label, w, h)

                if label == "LEFT":
                    left_landmarks = hand_landmarks.landmark
                elif label == "RIGHT":
                    right_landmarks = hand_landmarks.landmark

        # ── Color picker mode: Right hand 4-finger gesture ──
        if right_landmarks and state["main_mode"] == "DRAWING":
            gesture_4_detected = is_gesture_4(right_landmarks)
            if gesture_4_detected:
                state["color_picker_hold"] += 1
                if state["color_picker_hold"] == 10 and not state["color_picker_pressed"]:
                    # Toggle color picker mode
                    if state["sub_mode"] == "COLOR_PICKER":
                        state["sub_mode"] = "FREE"  # Exit color picker
                    else:
                        state["sub_mode"] = "COLOR_PICKER"  # Enter color picker
                    state["color_picker_pressed"] = True
            else:
                state["color_picker_hold"] = 0
                state["color_picker_pressed"] = False
        
        # ── Menu logic: Right hand OK gesture to toggle menu ──
        # (Skip OK gesture check when in COLOR_PICKER mode)
        if right_landmarks and not (state["main_mode"] == "DRAWING" and state["sub_mode"] == "COLOR_PICKER"):
            ok_detected = is_ok_gesture(right_landmarks, w, h)
            if ok_detected:
                state["ok_gesture_hold"] += 1
                # Toggle menu after 10 frames of holding gesture (faster response)
                # Only trigger once per gesture hold using the pressed flag
                if state["ok_gesture_hold"] == 10 and not state["ok_gesture_pressed"]:
                    if not state["menu_open"]:
                        # Open menu
                        state["menu_open"] = True
                        state["selected_mode"] = None  # Reset selection
                    else:
                        # Menu is open
                        if state["selected_mode"]:
                            # Confirm selection and activate mode
                            state["main_mode"] = state["selected_mode"]
                            state["sub_mode"] = "FREE"  # Always start in FREE mode
                            state["menu_open"] = False
                            state["selected_mode"] = None
                        else:
                            # Close menu without selection
                            state["menu_open"] = False
                    state["ok_gesture_pressed"] = True  # Mark gesture as processed
            else:
                # Gesture released
                state["ok_gesture_hold"] = 0
                state["ok_gesture_pressed"] = False  # Reset for next gesture

        # ── Menu mode: Left hand gestures to select main mode ──
        if state["menu_open"] and left_landmarks:
            gesture_1_detected = is_gesture_1(left_landmarks)
            gesture_2_detected = is_gesture_2(left_landmarks)
            
            if gesture_1_detected:
                state["gesture_1_hold"] += 1
                # Activate after 8 frames (smoother, faster)
                if state["gesture_1_hold"] == 8:
                    state["selected_mode"] = "DRAWING"  # Just mark selection
                    state["gesture_1_hold"] = 0
            else:
                state["gesture_1_hold"] = 0
            
            if gesture_2_detected:
                state["gesture_2_hold"] += 1
                # Activate after 8 frames (smoother, faster)
                if state["gesture_2_hold"] == 8:
                    state["selected_mode"] = "INTERACTIVE"  # Just mark selection
                    state["gesture_2_hold"] = 0
            else:
                state["gesture_2_hold"] = 0

        # ── Drawing mode logic ──
        if state["main_mode"] == "DRAWING" and not state["menu_open"]:
            # Right hand controls sub-modes (when not in COLOR_PICKER)
            if right_landmarks and state["sub_mode"] != "COLOR_PICKER":
                lm = right_landmarks
                index  = lm[8].y  < lm[6].y
                middle = lm[12].y < lm[10].y
                ring   = lm[16].y < lm[14].y
                pinky  = lm[20].y < lm[18].y

                if index and not middle and not ring and not pinky:
                    state["sub_mode"] = "DRAW"
                elif index and middle and not ring and not pinky:
                    state["sub_mode"] = "FREE"
                elif index and middle and ring and not pinky:
                    state["sub_mode"] = "ERASE"

            # Left hand controls drawing
            if left_landmarks:
                if state["sub_mode"] == "COLOR_PICKER":
                    # Color picker mode - just display palette, no drawing
                    state["prev_point"] = None
                elif state["sub_mode"] == "DRAW":
                    if only_index_up(left_landmarks):
                        draw_point = get_pos(left_landmarks[8], w, h)
                        current_color = DRAW_COLORS[state["draw_color_idx"]]
                        
                        if state["prev_point"]:
                            cv2.line(canvas, state["prev_point"], draw_point, current_color, DRAW_THICKNESS)
                        state["prev_point"] = draw_point
                        # Pen indicator with selected color
                        cv2.circle(frame, draw_point, 12, (0, 150, 0), 2)
                        cv2.circle(frame, draw_point, 8, current_color, -1)
                        cv2.circle(frame, draw_point, 4, (255, 255, 255), -1)
                    else:
                        state["prev_point"] = None
                elif state["sub_mode"] == "FREE":
                    state["prev_point"] = None
                elif state["sub_mode"] == "ERASE":
                    # Check if hand is fully open for BIG ERASER
                    if is_open_hand(left_landmarks):
                        # Get palm center and calculate radius reaching just before fingertips
                        center_pos = hand_center(left_landmarks, w, h)
                        
                        # 0.95 multiplier makes it almost as big as the full hand reach
                        dynamic_radius = int(hand_size(left_landmarks, w, h) * 0.95)
                        
                        if state["prev_point"]:
                            # Draw thick line for massive erasing based on dynamic radius
                            cv2.line(canvas, state["prev_point"], center_pos, (0, 0, 0), dynamic_radius * 2)
                        
                        state["prev_point"] = center_pos
                        
                        # Big Eraser visual feedback (Circle overlay)
                        cv2.circle(frame, center_pos, dynamic_radius, (100, 100, 255), 2)
                        cv2.circle(frame, center_pos, 5, (0, 0, 255), -1) # Center indicator
                        
                    # Precision eraser using only the index finger
                    elif only_index_up(left_landmarks):
                        draw_point = get_pos(left_landmarks[8], w, h)
                        if state["prev_point"]:
                            cv2.line(canvas, state["prev_point"], draw_point, (0, 0, 0), ERASER_SIZE)
                        
                        state["prev_point"] = draw_point
                        
                        # Precision eraser visual feedback
                        cv2.circle(frame, draw_point, ERASER_SIZE // 2 + 3, (100, 100, 255), 2)
                    else:
                        state["prev_point"] = None


        
        else:
            state["prev_point"] = None

        # Merge canvas
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        frame[mask > 0] = canvas[mask > 0]

        # ── Draw palette if in COLOR_PICKER mode ──
        if state["main_mode"] == "DRAWING" and state["sub_mode"] == "COLOR_PICKER":
            palette_center = (WIDTH // 2, HEIGHT - 150)
            draw_circular_color_palette(frame, palette_center, radius=120, inner_radius=60)

        # ── HUD with Enhanced Styling ──
        if state["main_mode"] == "DRAWING":
            mode_colors = {"DRAW": (0, 255, 100), "FREE": (100, 200, 255), "ERASE": (0, 150, 255), "COLOR_PICKER": (255, 100, 0)}
            mode_bg_colors = {"DRAW": (20, 80, 20), "FREE": (20, 60, 100), "ERASE": (20, 60, 100), "COLOR_PICKER": (80, 60, 20)}
            
            # Mode indicator with background panel
            cv2.rectangle(frame, (8, 8), (320, 65), mode_bg_colors[state['sub_mode']], -1)
            cv2.rectangle(frame, (8, 8), (320, 65), mode_colors[state['sub_mode']], 3)
            # Mode text shadow
            cv2.putText(frame, f"MODE: {state['sub_mode']}", (13, 42),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (20, 20, 20), 3)
            cv2.putText(frame, f"MODE: {state['sub_mode']}", (12, 41),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, mode_colors[state["sub_mode"]], 2)

        # Path display with enhanced styling
        display_path = state["screenshots_dir"]
        if len(display_path) > 50: display_path = "..." + display_path[-47:]
        cv2.rectangle(frame, (8, 68), (720, 98), (25, 25, 25), -1)
        cv2.rectangle(frame, (8, 68), (720, 98), (100, 100, 100), 2)
        cv2.putText(frame, f"Save: {display_path}", (14, 88), cv2.FONT_HERSHEY_DUPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(frame, f"Save: {display_path}", (13, 87), cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)

        # Main mode indicator
        main_mode_color = (0, 255, 150) if state["main_mode"] == "DRAWING" else (255, 150, 0)
        cv2.rectangle(frame, (8, 100), (280, 128), (30, 30, 30), -1)
        cv2.rectangle(frame, (8, 100), (280, 128), main_mode_color, 2)
        cv2.putText(frame, f"MAIN: {state['main_mode']}", (15, 120), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, main_mode_color, 2)

        # Controls text with background
        if state["main_mode"] == "DRAWING":
            if state["sub_mode"] == "COLOR_PICKER":
                controls_text = "COLOR PICKER MODE | RIGHT: 4 fingers=Exit  |  S=Snap | D=Drawing | Q=Quit"
            else:
                controls_text = "RIGHT: 1 finger=DRAW | 2 fingers=FREE | 3 fingers=ERASE | 4 fingers=COLOR PALETTE | Pinch=Menu | S=Snap | D=Drawing | Q=Quit"
        else:
            controls_text = "Interactive mode | RIGHT: Pinch=Menu  |  1/2=Switch modes  |  Q=Quit"
        cv2.rectangle(frame, (8, h - 28), (w - 8, h - 2), (25, 25, 25), -1)
        cv2.rectangle(frame, (8, h - 28), (w - 8, h - 2), (80, 80, 80), 2)
        cv2.putText(frame, controls_text, (14, h - 10), cv2.FONT_HERSHEY_DUPLEX, 0.48, (80, 80, 80), 2)
        cv2.putText(frame, controls_text, (13, h - 11), cv2.FONT_HERSHEY_DUPLEX, 0.48, (200, 200, 200), 1)

        # Settings button interaction
        mx, my  = state["mouse_pos"]
        hovered = BTN_X1 <= mx <= BTN_X2 and BTN_Y1 <= my <= BTN_Y2
        if hovered and state["btn_clicked"]:
            state["show_ui"] = True
            state["btn_clicked"] = False
        draw_settings_button(frame, hover=hovered)
        
        # Draw menu if open
        if state["menu_open"]:
            draw_main_menu(frame)

    else:
        # ── UI MODE ──
        draw_settings_ui(frame)

    # Save confirmation flash with enhanced effect
    if state["save_flash"] > 0:
        overlay = frame.copy()
        # Create gradient flash effect
        alpha = state["save_flash"] / 15
        flash_intensity = int(255 * alpha * 0.5)
        cv2.rectangle(overlay, (0, 0), (w, h), (150, 255, 150), -1)
        cv2.addWeighted(overlay, alpha * 0.35, frame, 1 - alpha * 0.35, 0, frame)
        
        # Get text size for proper centering
        text = "SAVED!"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 2.0
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Rectangle dimensions and position - centered
        rect_padding = 40
        rect_width = text_size[0] + rect_padding * 2
        rect_height = text_size[1] + rect_padding * 2
        rect_x1 = w // 2 - rect_width // 2
        rect_y1 = h // 2 - rect_height // 2
        rect_x2 = rect_x1 + rect_width
        rect_y2 = rect_y1 + rect_height
        
        # Draw rectangle with proper framing
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), 3)
        
        # Place text centered in rectangle with proper alignment
        text_x = w // 2 - text_size[0] // 2
        text_y = h // 2 + text_size[1] // 2
        
        # SAVED! text with shadow and glow
        cv2.putText(frame, text, (text_x + 2, text_y + 2), 
                    font, font_scale, (50, 50, 50), thickness + 1)
        cv2.putText(frame, text, (text_x, text_y), 
                    font, font_scale, (0, 255, 0), thickness)
        state["save_flash"] -= 1

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'): canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    elif key == ord('s'):
        save_screenshot(frame, canvas, False)
        state["save_flash"] = 15
    elif key == ord('d'):
        save_screenshot(frame, canvas, True)
        state["save_flash"] = 15
    elif key == ord('1'):  # Mark DRAWING mode as selected (opens menu if closed)
        if not state["menu_open"]:
            state["menu_open"] = True
            state["selected_mode"] = None
        state["selected_mode"] = "DRAWING"
    elif key == ord('2'):  # Mark INTERACTIVE mode as selected (opens menu if closed)
        if not state["menu_open"]:
            state["menu_open"] = True
            state["selected_mode"] = None
        state["selected_mode"] = "INTERACTIVE"

cap.release()
hands.close()
cv2.destroyAllWindows()