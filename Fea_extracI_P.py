import cv2
import os

import imagePreprocessingUtils as ipu
import settings

def isl4():
    labels=['I','J','K','L','M','N','O','P']

    for label in labels:

        for (subdirpath,subdirnames,images) in os.walk(ipu.PATH+'/'+label+'/'):
            count=0
            train_features=[]
            test_features=[]
            print('label {} is starting..'.format(label))
            ctr=0
            for image in images:
                imagePath=ipu.PATH+'/'+label+'/'+image
                img=cv2.imread(imagePath)
                if img is not None:
                    img=ipu.get_canny_edge(img,ctr,label)[0]
                    ctr+=1
                    surf_disc=ipu.get_SURF_descriptors(img,ctr,label)
                    if(count<(ipu.TOTAL_IMAGES*ipu.TRAIN_FACTOR*0.01)):

                        settings.train_img_disc.append(surf_disc)
                        settings.all_train_dis.extend(surf_disc)
                        settings.train_labels.append(settings.label_value)

                    elif((count>=(ipu.TOTAL_IMAGES*ipu.TRAIN_FACTOR*0.01)) and count <ipu.TOTAL_IMAGES):
                            
                        settings.test_img_disc.append(surf_disc)
                        settings.test_labels.append(settings.label_value)
                        # settings.test_labels.append
                    count+=1
            print('label {} is done'.format(label))
        settings.label_value+=1
# print(settings.label_value)
# print(len(settings.train_img_disc))
