# Kurulum (terminalde bir kere): pip install ultralytics opencv-python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")   # hazır eğitilmiş küçük model
INEK = 19                    # modelin listesinde "cow" sınıfının numarası

for sonuc in model("ahir_videosu.mp4", stream=True):
    inek_sayisi = sum(1 for sinif in sonuc.boxes.cls if int(sinif) == INEK)
    kare = sonuc.plot()      # bulunan nesnelerin etrafına kutu çizer
    cv2.putText(kare, f"Inek sayisi: {inek_sayisi}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Ahir", kare)
    if cv2.waitKey(1) & 0xFF == ord("q"):   # q'ya basınca çıkar
        break

cv2.destroyAllWindows()
