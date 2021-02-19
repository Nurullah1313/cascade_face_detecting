import cv2
import numpy as np

#haarcascade yöntemi ile birçok yüz ile eğitilmiş xml dosyası
yuzcascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)# wepcam
while True:
    _, frame = cam.read()#cameradan görüntüleri okuyoruz
    frame = cv2.flip(frame, 1)#görüntümüz aynalanmış bir şekilde geldiği için görüntüyü tekrar aynalar
    
    #işlem gücünü azaltmak açısından gri skalaya çevirmek önemlidir (ayrıca zorlamaya gerek yok)
    gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cascade dosyamız ile gri görüntü üzerinde yüzleri arar
    #scaleFactor ,minNeighbors argumanlarını kullanarak daha net sonuçlara ulaşılabilir
    yuzler = yuzcascade.detectMultiScale(gri,
                                  scaleFactor=1.2,
                                  minNeighbors=5,
                                  minSize=(10,25))

    #yukarda bulduğumuz değişkenler görüntü üzerinde bulunan
    # yüzlerin sol üst köşesinin kordinatlarını ayrıca bu yüzün
    # genişlik ve yüksekliğini verir
    #bu değerleri kullanarak frame üzerinde yuzun çevresini çizer
    for x,y,w,h in yuzler:
        cv2.rectangle(frame,(x,y),(x+w, y+h), (0, 255, 0), 2)
    
    #üzerinde yuzun bulunduğu bölgeyi kare içerisine alınmış görüntüyü ekrana verir
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.imwrite('capture.png',frame)
        break

cam.release()
cv2.destroyAllWindows()
