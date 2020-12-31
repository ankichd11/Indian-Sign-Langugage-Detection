
import numpy as np
import cv2
import os
import pickle
import imagePreprocessingUtils as ipu


CAPTURE_FLAG = False

class_labels = ipu.get_labels()



def recognise(cluster_model, classify_model):
    global CAPTURE_FLAG
    gestures = ipu.get_all_gestures()
    cv2.imwrite("all_gestures.jpg", gestures)
    camera = cv2.VideoCapture(0)
    print('Now camera window will be open, then \n1) Place your hand gesture in ROI (rectangle) \n2) Press esc key to exit.')
    count = 0
    while(True):
        (t,frame) = camera.read()
        frame = cv2.flip(frame,1)
        cv2.rectangle(frame,ipu.START, ipu.END,(0,255,0),2 )
        cv2.imshow("All_gestures", gestures)
        pressedKey = cv2.waitKey(1)
        if pressedKey == 27:
            break
        elif pressedKey == ord('p'):
            if(CAPTURE_FLAG):
                CAPTURE_FLAG = False
            else:
                CAPTURE_FLAG = True
        if(CAPTURE_FLAG):
            # Region of Interest
            roi = frame[ ipu.START[1]+5:ipu.END[1], ipu.START[0]+5:ipu.END[0]]
            if roi is not None:
                roi = cv2.resize(roi, (ipu.IMG_SIZE,ipu.IMG_SIZE))
                img = ipu.get_canny_edge(roi,1,'1')[0]
                cv2.imshow("Edges ",img)
                print(img)
                surf_disc = ipu.get_SURF_descriptors(img,1,'1')
            print(type(surf_disc))
            if surf_disc is not None:
                visual_words = cluster_model.predict(surf_disc)
                print('visual words collected.')
                bovw_histogram = np.array(np.bincount(visual_words, minlength=ipu.N_CLASSES * ipu.CLUSTER_FACTOR))
                pred = classify_model.predict([bovw_histogram])
                label = class_labels[pred[0]]
                rectangle_bgr = (150,150,150)
                (text_width, text_height) = cv2.getTextSize('Predicted text:      ', 1, fontScale=1.5, thickness=2)[0]
                # set the text start position
                text_offset_x = 5
                text_offset_y = 80
                # make the coords of the box with a small padding of two pixels
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width+30 , text_offset_y + text_height +50))
                cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
                frame = cv2.putText(frame, 'Predicted text: ', (8,120), cv2.FONT_HERSHEY_COMPLEX,1, (0,0,0), 2, cv2.LINE_AA)
                frame = cv2.putText(frame, label, (260,130), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
                
        cv2.imshow("Video",frame)
    camera.release()
    cv2.destroyAllWindows()



clustering_model = pickle.load(open('mini_kmeans_model.sav', 'rb'))    
classification_model = pickle.load(open('svm_model.sav', 'rb'))
recognise(clustering_model,classification_model)

