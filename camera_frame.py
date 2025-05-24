import cv2

def capture_frame_from_camera(camera_url):
    # Add options to improve RTSP stream handling
    cap = cv2.VideoCapture(
        camera_url,
        cv2.CAP_FFMPEG
    )
    if not cap.isOpened():
        print("Error: Could not open video stream from camera.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame from camera.")
        return None

    ret, jpeg = cv2.imencode('.jpg', frame)
    if not ret:
        print("Error: Could not encode frame to JPEG.")
        return None

    return jpeg.tobytes()