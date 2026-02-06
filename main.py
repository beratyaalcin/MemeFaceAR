import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands #el takibi modulu
mp_draw = mp.solutions.drawing_utils #cizim modulu
mp_face = mp.solutions.face_detection #yuz tespiti modulu

#modul hassasiyet oranlari
hands = mp_hands.Hands(min_detection_confidence=0.7)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)
monkey_img = cv2.imread("monkey.png")

cap = cv2.VideoCapture(0)
print("Sistem hazır, kamera açılıyor...")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1) #aynalama
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #opencv blue-green-red seklinde okurken mediapipe red-green-blue seklinde okur bunu ayarlama kodu
    
    hand_res = hands.process(rgb_img)
    face_res = face_detection.process(rgb_img)
    
    if hand_res.multi_hand_landmarks:
        for landmarks in hand_res.multi_hand_landmarks:
            #el iskeleti ciz
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            #isaret parmagi (☝️) tespiti
            #landmark 8: isaret parmaginin ucu
            #landmark 6: isaret parmaginin ikinci eklemi
            #mantigi: eger uc noktasinin y degeri, eklem noktasından kucukse y8 < y6 
            #bu parmagin yukari kalktigi anlamina geliyor.
            if landmarks.landmark[8].y < landmarks.landmark[6].y:
                if face_res.detections:
                    for detection in face_res.detections:
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                        bw, bh = int(bbox.width * w), int(bbox.height * h)
                        try:
                            #monkey.png dosyasini yuzun boyutuna gore yeniden boyutlandirip
                            #videodaki yuz koordinatlari uzerine overlay yapiyoruz
                            m_res = cv2.resize(monkey_img, (bw, bh))
                            frame[max(0,y):y+bh, max(0,x):x+bw] = m_res
                        except: pass

    cv2.imshow('MemeFace AR - Setup', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()