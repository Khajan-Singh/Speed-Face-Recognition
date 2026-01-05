import cv2
import mediapipe as mp
import numpy as np
import time


REF_IMAGE_PATH = "speed.jpg"   # uploaded face image (special face)
MATCH_THRESHOLD = 100          # tweak based on distances
DELAY_SECONDS = 0.25            # delay before overlay appears

mp_face_mesh = mp.solutions.face_mesh


def dist_px(landmarks, i, j, w, h):
    """Pixel distance between two landmark indices."""
    p1 = landmarks[i]
    p2 = landmarks[j]
    dx = (p1.x - p2.x) * w
    dy = (p1.y - p2.y) * h
    return np.hypot(dx, dy)


def get_expression_vector(image, mesh):
    """
    Returns a 3D vector:
        [left_eye_EAR, right_eye_EAR, mouth_ratio]
    or None if no face detected.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image.shape

    #Left eye
    left_vert = dist_px(landmarks, 159, 145, w, h)
    left_horiz = dist_px(landmarks, 33, 133, w, h)
    left_EAR = left_vert / (left_horiz + 1e-6)

    #Right eye
    right_vert = dist_px(landmarks, 386, 374, w, h)
    right_horiz = dist_px(landmarks, 362, 263, w, h)
    right_EAR = right_vert / (right_horiz + 1e-6)

    #Mouth
    mouth_vert = dist_px(landmarks, 13, 14, w, h)
    mouth_horiz = dist_px(landmarks, 78, 308, w, h)
    mouth_ratio = mouth_horiz / (mouth_vert + 1e-6)

    # Feature vector
    return np.array([left_EAR, right_EAR, mouth_ratio], dtype=np.float32)


# 1. LOAD REFERENCE IMAGE & VECTOR
ref_img = cv2.imread(REF_IMAGE_PATH)
if ref_img is None:
    print(f"Error: Could not load reference image '{REF_IMAGE_PATH}'")
    raise SystemExit

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
) as ref_mesh:
    ref_vec = get_expression_vector(ref_img, ref_mesh)

if ref_vec is None:
    print("Error: No face detected in reference image.")
    raise SystemExit

print("Reference expression vector:", ref_vec)
print("Reference loaded. Starting webcam...")


# 2. START WEBCAM
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

# For delayed overlay
first_match_time = None
overlay_active = False

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as live_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        live_vec = get_expression_vector(frame, live_mesh)

        status = "No face detected"
        distance_text = "N/A"
        is_match_now = False

        if live_vec is not None:
            distance = float(np.linalg.norm(ref_vec - live_vec))
            distance_text = f"{distance:.4f}"
            print("Live vec:", live_vec, "  dist:", distance)

            if distance < MATCH_THRESHOLD:
                is_match_now = True
                status = "MATCH (pending delay)..."
            else:
                status = "No match"

        # DELAY LOGIC
        now = time.time()

        if is_match_now:
            if first_match_time is None:
                # Just started matching; start the timer
                first_match_time = now
                overlay_active = False
            else:
                # Check if we've been matching long enough
                if (now - first_match_time) >= DELAY_SECONDS:
                    overlay_active = True
                    status = "MATCH!"
        else:
            # Reset everything when match is lost
            first_match_time = None
            overlay_active = False

        # FULL SCREEN OVERLAY AFTER DELAY
        if overlay_active:
            h, w, _ = frame.shape
            overlay_full = cv2.resize(ref_img, (w, h))
            frame = overlay_full   # FULL REPLACEMENT

        # Draw status text on top
        cv2.putText(
            frame,
            f"{status}  dist={distance_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0) if overlay_active else (0, 0, 255),
            2
        )

        cv2.imshow("Expression Matcher (press Q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
