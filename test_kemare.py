import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Tidak bisa membuka kamera")
else:
    print("Kamera berhasil dibuka")
    ret, frame = cap.read()
    if ret:
        print("Berhasil membaca frame")
        cv2.imshow('Test Kamera', frame)
        cv2.waitKey(0)
    else:
        print("Tidak bisa membaca frame")
cap.release()
cv2.destroyAllWindows()