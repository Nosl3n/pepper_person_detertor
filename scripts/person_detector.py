#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os

# Ruta a los clasificadores de Haar
haar_cascade_path = '/usr/share/opencv/haarcascades/'

# Cargar clasificadores en cascada de Haar para rostros y cuerpos
face_cascade = cv2.CascadeClassifier(os.path.join(haar_cascade_path, 'haarcascade_frontalface_default.xml'))
body_cascade = cv2.CascadeClassifier(os.path.join(haar_cascade_path, 'haarcascade_fullbody.xml'))
upper_body_cascade = cv2.CascadeClassifier(os.path.join(haar_cascade_path, 'haarcascade_upperbody.xml'))
lower_body_cascade = cv2.CascadeClassifier(os.path.join(haar_cascade_path, 'haarcascade_lowerbody.xml'))

if face_cascade.empty() or body_cascade.empty() or upper_body_cascade.empty() or lower_body_cascade.empty():
    raise IOError("Failed to load Haar cascades. Check the paths to the cascade files.")

def detect_and_display(frame):
    person_detected = False

    # Detectar personas usando HOG
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        person_detected = True

    # Detectar rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        person_detected = True

    # Detectar cuerpos completos
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        person_detected = True

    # Detectar torsos
    upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in upper_bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        person_detected = True

    # Detectar piernas
    lower_bodies = lower_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in lower_bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        person_detected = True

    # Mostrar la imagen con las detecciones
    cv2.imshow('Person Detection', frame)
    cv2.waitKey(3)

    return person_detected

def image_callback(image_msg):
    try:
        # Convertir el mensaje de imagen de ROS a OpenCV
        cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

    # Detectar y mostrar personas en la imagen
    person_detected = detect_and_display(cv_image)
    if person_detected:
        rospy.loginfo("Persona detectada")
    else:
        rospy.loginfo("No hay personas")

if __name__ == '__main__':
    rospy.init_node('person_detector_node', anonymous=True)
    rospy.Subscriber('/pepper_robot/camera/front/camera/image_raw', Image, image_callback)
    bridge = CvBridge()

    rospy.loginfo("Nodo de detección de personas iniciado.")
    rospy.spin()

    # Cerrar todas las ventanas de OpenCV
    cv2.destroyAllWindows()
