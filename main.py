import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe import tasks

# Access vision module
vision = tasks.vision

# Filter pairs
pairs = {
    "middle": ("SEPIA", "NEGATIVE"),
    "ring": ("BLUR", "GLITCH"),
    "pinky": ("EDGE", "CARTOON")
}

# Toggle state
st = {k: 0 for k in pairs}

# Current filter
cur = "SEPIA"

# Timing / thresholds
DEB = 0.6
CAP = 1.2
TT = 45
TP = 35

# Last action time
la = 0
lc = 0

pinch_on = False

MAIN = "Gesture-Controlled Photo App"
POP = "Captured (ESC / Close to resume)"

paused = False
freeze = None

# Sepia matrix
SEPIA_M = np.array([
    [0.272, 0.534, 0.131],
    [0.349, 0.686, 0.168],
    [0.393, 0.769, 0.189]
])

# Landmark indices
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

ids = {
    "thumb": THUMB_TIP,
    "index": INDEX_TIP,
    "middle": MIDDLE_TIP,
    "ring": RING_TIP,
    "pinky": PINKY_TIP,
}


# Apply filters
def apply(img, t):

    if t == "SEPIA":
        return np.clip(
            cv2.transform(img, SEPIA_M),
            0,
            255
        ).astype(np.uint8)

    if t == "NEGATIVE":
        return cv2.bitwise_not(img)

    if t == "BLUR":
        return cv2.GaussianBlur(img, (15, 15), 0)

    if t == "GLITCH":

        h, w = img.shape[:2]

        r = img[:, :, 2]
        g = img[:, :, 1]
        b = img[:, :, 0]

        return cv2.merge([
            np.roll(b, -int(0.02 * w), axis=1),
            g,
            np.roll(r, int(0.04 * w), axis=1)
        ])

    if t == "EDGE":

        return cv2.Canny(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            80,
            160
        )

    if t == "CARTOON":

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.adaptiveThreshold(
            cv2.medianBlur(gray, 7),
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            2
        )

        color = cv2.bilateralFilter(img, 9, 75, 75)

        return cv2.bitwise_and(color, color, mask=edges)

    return img


# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

# -----------------------------
# MediaPipe HandLandmarker
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

try:

    landmarker = HandLandmarker.create_from_options(options)

    print("✅ HandLandmarker initialized")

except Exception as e:

    print("❌ Initialization Error:", e)
    exit()

# Create window
cv2.namedWindow(MAIN, cv2.WINDOW_NORMAL)

print("✅ App Started")
print("Show your hand to camera")
print("Current filter:", cur)

frame_count = 0
detection_frames = 0

while True:

    # Pause mode
    if paused:

        cv2.imshow(MAIN, freeze)

        key = cv2.waitKey(50) & 0xFF

        if key == ord("q"):
            break

        if key == 27:

            paused = False
            pinch_on = False

            try:
                cv2.destroyWindow(POP)
            except:
                pass

            continue

        try:

            if cv2.getWindowProperty(
                POP,
                cv2.WND_PROP_VISIBLE
            ) <= 0:

                paused = False
                pinch_on = False

        except:
            paused = False
            pinch_on = False

        continue

    # Read frame
    ok, img = cap.read()

    if not ok:
        print("❌ Frame read failed")
        break

    frame_count += 1

    # Mirror effect
    img = cv2.flip(img, 1)

    h, w = img.shape[:2]

    now = time.time()

    capture = False

    # Convert to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    try:

        detection_result = landmarker.detect(mp_image)

        if (
            detection_result and
            detection_result.hand_landmarks and
            len(detection_result.hand_landmarks) > 0
        ):

            detection_frames += 1

            hand_landmarks = detection_result.hand_landmarks[0]

            # Draw landmarks
            for landmark in hand_landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    img,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

            # Hand connections
            HAND_CONNECTIONS = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (5,9),(9,10),(10,11),(11,12),
                (9,13),(13,14),(14,15),(15,16),
                (13,17),(17,18),(18,19),(19,20),
                (0,17)
            ]

            for start, end in HAND_CONNECTIONS:

                start_pos = (
                    int(hand_landmarks[start].x * w),
                    int(hand_landmarks[start].y * h)
                )

                end_pos = (
                    int(hand_landmarks[end].x * w),
                    int(hand_landmarks[end].y * h)
                )

                cv2.line(
                    img,
                    start_pos,
                    end_pos,
                    (0, 255, 0),
                    2
                )

            # Fingertips
            tips = {}

            for name, idx in ids.items():

                lm = hand_landmarks[idx]

                tips[name] = (
                    int(lm.x * w),
                    int(lm.y * h)
                )

            # Thumb + index
            tx, ty = tips["thumb"]
            ix, iy = tips["index"]

            # Pinch distance
            pinch_dist = (
                ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5
            )

            pinch = pinch_dist < TP

            # Capture
            if (
                pinch and
                not pinch_on and
                now - lc > CAP
            ):

                pinch_on = True
                capture = True
                lc = now

                print("📸 Capture Triggered")

            if not pinch:
                pinch_on = False

            # Filter gestures
            if not pinch:

                for finger in pairs:

                    fx, fy = tips[finger]

                    touch_dist = (
                        ((tx - fx) ** 2 +
                         (ty - fy) ** 2) ** 0.5
                    )

                    if (
                        touch_dist < TT and
                        now - la > DEB
                    ):

                        cur = pairs[finger][st[finger]]

                        st[finger] ^= 1

                        la = now

                        print("🎨 Filter:", cur)

                        break

    except Exception as e:

        if frame_count % 60 == 0:
            print("Detection Error:", e)

    # Apply filter
    out = apply(img, cur)

    # Convert edge back to BGR
    if cur == "EDGE":
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    # UI Text
    cv2.putText(
        out,
        f"Filter: {cur}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        out,
        f"Frames: {frame_count} | Hands: {detection_frames}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Save image
    if capture:

        name = f"picture_{int(now)}.jpg"

        cv2.imwrite(name, out)

        print(f"✅ Saved: {name}")

        paused = True

        freeze = out.copy()

        cv2.imshow(POP, freeze)

    # Show output
    cv2.imshow(MAIN, out)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()

cv2.destroyAllWindows()

print(
    f"✅ Closed | "
    f"Frames: {frame_count} | "
    f"Detected Hands: {detection_frames}"
)
