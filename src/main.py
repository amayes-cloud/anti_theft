import cv2
import time
import threading
from ultralytics import YOLO

model = YOLO("yolo26s.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Camera connection failed")
    exit()

latest_frame = None
running = True
lock = threading.Lock()

def capture():
    global latest_frame

    while running:
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = frame

thread = threading.Thread(target=capture, daemon=True)
thread.start()

last_annotated = None
frame_count = 0
ai_fps = 0
last_time = time.time()

while True:
    with lock:
        frame = latest_frame.copy() if latest_frame is not None else None

    if frame is None:
        continue

    frame_count += 1

    if frame_count % 2 == 0:
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.30,
            imgsz=416,
            device="cpu",
            classes=[0],
            verbose=False
        )

        last_annotated = results[0].plot()

        now = time.time()
        ai_fps = 1 / max(now - last_time, 0.001)
        last_time = now

    if last_annotated is not None:
        display = last_annotated.copy()
    else:
        display = frame.copy()

    cv2.putText(
        display,
        f"AI FPS: {ai_fps:.1f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Store Surveillance", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

running = False
thread.join(timeout=1)
cap.release()
cv2.destroyAllWindows()
