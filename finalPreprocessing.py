import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
import sklearn.metrics as skmetrics
import random
import pickle
import imagePreprocessingUtils as ipu
import sklearn as sk
import settings


def mini_kmeans(k, descriptor_list):
    print('Mini batch K-Means started.')
    print ('%i descriptors before clustering' % descriptor_list.shape[0])
    kmeans_model = MiniBatchKMeans(k)
    kmeans_model.fit(descriptor_list)
    print('Mini batch K means trained to get visual words.')
    filename = 'mini_kmeans_model.sav'
    # f=open(filename,'ab')
    pickle.dump(kmeans_model, open(filename,'wb'))
    return kmeans_model

def predict_svm(X_train, X_test, y_train, y_test):
    svc=SVC(kernel='linear') 
    print("SVM started.")
    svc.fit(X_train,y_train)
    filename = 'svm_model.sav'
    # f=open(filename,'ab')
    pickle.dump(svc,open(filename,'wb'))
    y_pred=svc.predict(X_test)
    # header = 'ImageId,PredictedLabel,TrueLabel'
    np.savetxt('submission_svm.csv', np.c_[range(1,len(y_test)+1),y_pred,y_test], delimiter=',', header = 'ImageId,PredictedLabel,TrueLabel',comments = '', fmt='%d')
    calculate_metrics("SVM",y_test,y_pred)


def calculate_metrics(method,label_test,label_pred):
    print("Accuracy score for ",method,skmetrics.accuracy_score(label_test,label_pred))
    print("Precision_score for ",method,skmetrics.precision_score(label_test,label_pred,average='micro'))
    print("f1 score for ",method,skmetrics.f1_score(label_test,label_pred,average='micro'))    


    print("Recall score for ",method,skmetrics.recall_score(label_test,label_pred,average='micro'))

def classify():
    mini_kmeans_model = mini_kmeans(ipu.N_CLASSES * ipu.CLUSTER_FACTOR, np.array(settings.all_train_dis))

    print('Collecting visual words for train .....')
    train_images_visual_words = [mini_kmeans_model.predict(visual_words) for visual_words in settings.train_img_disc]

    print('Visual words for train data collected. length is %i' % len(train_images_visual_words))

    print('Collecting visual words for test .....')
    test_images_visual_words = [mini_kmeans_model.predict(visual_words) for visual_words in settings.test_img_disc]
    print('Visual words for test data collected. length is %i' % len(test_images_visual_words))

    print('Calculating Histograms for train...')
    bovw_train_histograms = np.array([np.bincount(visual_words, minlength=ipu.N_CLASSES * ipu.CLUSTER_FACTOR) for visual_words in train_images_visual_words])
    print('Histograms for training data are collected. Length : %i ' % len(bovw_train_histograms))

    print('Calculating Histograms for test...')
    bovw_test_histograms = np.array([np.bincount(visual_words, minlength=ipu.N_CLASSES * ipu.CLUSTER_FACTOR) for visual_words in test_images_visual_words])
    print('Histograms for testing data are collected. Length : %i ' % len(bovw_test_histograms))

    print('Each histogram length is : %i' % len(bovw_train_histograms[0]))
        #----------------------
    print('============================================')



    X_train = bovw_train_histograms
    X_test = bovw_test_histograms
    Y_train = settings.train_labels
    Y_test = settings.test_labels

    # print("@")
    # print(len(X_train))
    # print(len(Y_train))
    # print("#")

    buffer  = list(zip(X_train, Y_train))
    random.shuffle(buffer)
    random.shuffle(buffer)
    random.shuffle(buffer)
    X_train, Y_train = zip(*buffer)


    buffer  = list(zip(X_test, Y_test))
    random.shuffle(buffer)
    random.shuffle(buffer)
    X_test, Y_test = zip(*buffer)

    print('Length of X-train:  %i ' % len(X_train))
    print('Length of Y-train:  %i ' % len(Y_train))
    print('Length of X-test:  %i ' % len(X_test))
    print('Length of Y-test:  %i ' % len(Y_test))

    predict_svm(X_train, X_test,Y_train, Y_test)
