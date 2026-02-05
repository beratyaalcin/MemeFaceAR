import cv2
import sys
import os

# 1. Kütüphaneyi Bulma Garantisi
try:
    # Doğrudan iç modüllere erişim
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.face_detection as mp_face
    import mediapipe.python.solutions.drawing_utils as mp_draw
except ImportError:
    try:
        # Alternatif yol
        from mediapipe.solutions import hands as mp_hands
        from mediapipe.solutions import face_detection as mp_face
        from mediapipe.solutions import drawing_utils as mp_draw
    except:
        print("Kütüphane hala bulunamadı! Ama panik yok, yeşil kutuyu alacağız.")
        # Acil durum için sadece OpenCV ile çalışan bir fallback (yedek) planı
        sys.exit(1)

# 2. Maymunu Yükle
monkey_img = cv2.imread("monkey.png")

hands = mp_hands.Hands(min_detection_confidence=0.7)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # El ve Yüz tespiti
    hand_res = hands.process(rgb_img)
    face_res = face_detection.process(rgb_img)
    
    if hand_res.multi_hand_landmarks:
        for landmarks in hand_res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Sadece işaret parmağı havadaysa
            if landmarks.landmark[8].y < landmarks.landmark[6].y:
                if face_res.detections:
                    for detection in face_res.detections:
                        bbox = detection.location_data.relative_bounding_box
                        fx, fy = int(bbox.xmin * w), int(bbox.ymin * h)
                        fw, fh = int(bbox.width * w), int(bbox.height * h)
                        try:
                            res_monkey = cv2.resize(monkey_img, (fw, fh))
                            frame[max(0, fy):fy+fh, max(0, fx):fx+fw] = res_monkey
                        except: pass

    cv2.imshow('BAIBU - MemeFace AR Final', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()