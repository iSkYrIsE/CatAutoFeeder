import cv2
import sys
import os
import time
import numpy as np
import RPi.GPIO as GPIO
import datetime as dt
import pigpio
from gpiozero import DistanceSensor

os.system("sudo pigpiod")
pi = pigpio.pi()

#-------------------------------------------

#Preparamos el sensor de ultrasonidos
sensor = DistanceSensor(echo=17, trigger=4)


cascPath = "haarcascade_frontalcatface.xml"

flag_abierto = 0
flag_tiempo = 0
flag_video = 0
var_tiempo = 0
var_tiempo_inicial = 0
hora = 0
GPIO.setwarnings(False)



while True:
    
    #Mostramos la distancia del sensor de ultrasonidos
    print(sensor.distance)
    pi.set_servo_pulsewidth(5, 1300)
    #El gato esta proximo al dispositivo, activamos camara
    if(sensor.distance<0.6):
        flag_video=1;
        faceCascade = cv2.CascadeClassifier(cascPath)
        video_capture = cv2.VideoCapture(0)
        #Mientras el gato este proximo recogemos señal de video
        while(sensor.distance<0.6):
            #Cargamos la hora del dia para comprobaciones posteriores
            hora = dt.datetime.today().hour
            
            #Preparamos el procesamiento de video
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
            #Comprobamos si hay rostros en el frame    
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Comprobamos si hay gatos en el frame
            # Si el gato permanece 6 segundos delante de la camara, abrimos el comedero   
            if(len(faces) and flag_abierto == 0):
                
                if(flag_abierto == 0):
                    img = cv2.imread("Imagenes/gato2.jpg")
                    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    faces2 = faceCascade.detectMultiScale(
                        gray2,
                        scaleFactor=1.02,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    crop_img2=0
                    crop_img=0
                
                    for (x,y,w,h) in faces:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            crop_img = frame[y:y+h, x:x+w]
                            
                    for (x,y,w,h) in faces2:
                            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            crop_img2 = img[y:y+h, x:x+w]
                        
                
#                crop_img2 = cv2.resize(crop_img2, (224, 224))
                
                    if crop_img.shape == crop_img2.shape:
                        print("The images have same size and channels")
                        difference = cv2.subtract(crop_img, crop_img2)
                        b, g, r = cv2.split(difference)
                     
                        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                            print("The images are completely Equal")
                        else:
                            print("The images are NOT equal")
                    
                    sift = cv2.xfeatures2d.SIFT_create()
                    kp_1, desc_1 = sift.detectAndCompute(crop_img, None)
                    kp_2, desc_2 = sift.detectAndCompute(crop_img2, None)
                     
                    index_params = dict(algorithm=0, trees=5)
                    search_params = dict()
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                     
                    matches = flann.knnMatch(desc_1, desc_2, k=2)
                
                
                    good_points = []
                    for m, n in matches:
                        if m.distance < 0.6*n.distance:
                            good_points.append(m)
                     
                    # Define how similar they are
                    number_keypoints = 0
                    if len(kp_1) <= len(kp_2):
                        number_keypoints = len(kp_1)
                    else:
                        number_keypoints = len(kp_2)
                    
                    
                    
                    print("Keypoints 1ST Image: " + str(len(kp_1)))
                    print("Keypoints 2ND Image: " + str(len(kp_2)))
                    print("GOOD Matches:", len(good_points))
                    print("How good it's the match: ", len(good_points) / number_keypoints * 100)
                     
                    result = cv2.drawMatches(crop_img, kp_1, crop_img2, kp_2, good_points, None)
                     
                     
#                    cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4))
                    cv2.imwrite("feature_matching.jpg", result)
                     
                     
#                    cv2.imshow("Original", cv2.resize(crop_img, None, fx=0.4, fy=0.4))
#                    cv2.imshow("Duplicate", cv2.resize(crop_img2, None, fx=0.4, fy=0.4))

                if(len(good_points) > 20):
                    print("Es el gato registrado")
                    flag_abierto = 1
                
                
                
                if flag_abierto == 1:

                    print("Hay gatos en el frame")
                    
#                    if(flag_tiempo == 0):
#                        flag_tiempo=1
#                        var_tiempo_inicial = time.time()+6
                    
#                    if(flag_tiempo == 1):
#                        print(var_tiempo)
#                        print(var_tiempo_inicial)
#                        var_tiempo = time.time()
                        
#                        if(var_tiempo > var_tiempo_inicial):
                    print("Abriendo comedero")
                            #Si es entre las 00:00 - 8:00 abrimos primera compuerta
                    if(hora>=0 and hora<16):
                        print("Primera posicion")
                        pi.set_servo_pulsewidth(5, 500)
#                        time.sleep(0.5)
                  
                    #Si es entre las 16:00 - 00:00 abrimos primera compuerta
                    if(hora>=16 and hora<23):
                        print("Segundo posicion")
                        pi.set_servo_pulsewidth(5, 2200)
                        time.sleep(0.5)
            
                else:
                    print("No hay gatos en el frame")
                
#                        flag_tiempo = 0
#                        var_timepo = 0
#                        pi.set_servo_pulsewidth(5, 500)
#                        time.sleep(0.5)
    
            for (x, y, w, h) in faces:
               cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Mostramos el resultado en el frame
#            cv2.imshow('Video', frame)

#            if cv2.waitKey(1) & 0xFF == ord('q'):
#                break
            
        
        
        
    if flag_video == 1:
        print("Saliendo...")
        video_capture.release()
        cv2.destroyAllWindows()
        flag_video=0;
        flag_abierto = 0
        time.sleep(2)
        pi.set_servo_pulsewidth(5, 1300)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


