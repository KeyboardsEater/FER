'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import tensorflow
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import model_from_json
import os
import cv2
from sklearn.metrics import confusion_matrix
import h5py
import matplotlib.pyplot as plt
import itertools



def getTrainValData():
    from shutil import copyfile

    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    newpath = '//Users/HP/Desktop/dissertation/code/Esmam2017-mscprojem-6c2a81260da0/dataFromPython3'
    alltrain = []
    allpredict = []
    for emotion in emotions:
            print(emotion)
            # path = '//Users/HP/Desktop/dissertation/code/Esmam2017-mscprojem-6c2a81260da0/NewAllImages/' #Face/' #sorted_set/'
            path = '../video_frames/'  # Face/' #sorted_set/'
            # pathJaffe = "//Users/emb24/Documents/PHd2/PhD/PycharmProjects/PivotHeadTest/jaffe/"
            # path = pathJaffe
            files = glob.glob(path + "%s/*" % emotion)# '%s' % emotion or  '%d' % (1,2)
            # random.shuffle(files)
            training = files[:int(len(files) * 0.80)]  # get first 80% of file list

            prediction = files[-int(len(files) * 0.20):]  # get last 20% of file list
            for t in training :
                fileonly = t.split('/')
                frame = cv2.imread(t)

                #frame = cv2.resize(frame, (150, 150))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                #clahe_image = clahe.apply(gray)
                clahe_image = gray
                rects = detector(clahe_image, 1)
                for rect in rects:
                    face = 1
                    # extract the ROI of the *original* face, then align the face
                    # using facial landmarks
                    (x, y, w, h) = rect_to_bb(rect)
                    d = rect

                    img = clahe_image  # np.float32(frame) / 255.0;
                    crop = img[d.top():d.bottom(), d.left():d.right()]
                #crop = cv2.resize(crop, (250, 250))

                cv2.imwrite(newpath+'/train/'+emotion+'/'+str(fileonly[-1]) ,  crop)
                #copyfile(t, newpath+'/train/'+emotion+'/'+str(fileonly[-1]))
            for p in prediction:
                fileonly = p.split('/')
                frame = cv2.imread(p)

                #frame = cv2.resize(frame, (250, 250))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                clahe_image = gray # clahe.apply(gray)
                rects = detector(clahe_image, 1)

                for rect in rects:
                    face = 1
                    # extract the ROI of the *original* face, then align the face
                    # using facial landmarks
                    (x, y, w, h) = rect_to_bb(rect)
                    d = rect

                    img = clahe_image  # np.float32(frame) / 255.0;
                    crop = img[d.top():d.bottom(), d.left():d.right()]
                #crop = cv2.resize(crop, (250, 250))

                cv2.imwrite(newpath + '/validation/' + emotion + '/' + str(fileonly[-1]) , crop)

def CNNSmall():

    # dimensions of our images.
    img_width, img_height = 150, 150

    train_data_dir = 'dataFromPython2/train'
    validation_data_dir = 'dataFromPython2/validation'
    nb_train_samples =2290 #168 #2290 #168 #2318 # 506 #2318 #362
    nb_validation_samples= 770 #41# 770#41 #=771 # 124# 771 #246
    epochs = 50
    batch_size =10# 100 # 16

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(124, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dropout(0.5))
    model.add(Dense(7))
    #model.add(Activation('sigmoid'))
    model.add(Activation('softmax'))

    '''model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])'''

    '''model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])'''

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical') # binary

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical') #binary

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    #model.save_weights('first_try.h5')


    # serialize model to JSON
    dir = dir = '//Users/HP/Desktop/dissertation/code/Esmam2017-mscprojem-6c2a81260da0'# '//Users/emb24/PycharmProjects/EmotionFacialUnitsVideo/'
    model_json = model.to_json()
    with open("SmallCNNADAM1.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("SMallCNNWeightADAM1.h5")
    print("Saved model to disk")

if __name__ == '__main__':
    #
    CNNSmall()
    dir = '//Users/HP/Desktop/dissertation/code/Esmam2017-mscprojem-6c2a81260da0'
    #classify each image in the validation set
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

    json_file = open( 'SmallCNNADAM1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights( "SMallCNNWeightADAM1.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #score = loaded_model.evaluate(X_test, y_test, verbose=0)
    path = 'dataFromPython2/validation/'
    Accurate = 0
    counter = 0
    labelsValues = []
    PredictedValues = []
    for emotion in emotions:
        for file in os.listdir(path+emotion):
            if 'png' in file:

                image = cv2.imread(path+emotion+'/'+file)
                image = cv2.resize(image, (150, 150))
                image = image.reshape(1, len(image), len(image),3)
                label = emotions.index(emotion)
                pred = loaded_model.predict(image)
                pred1 = list(pred[0])
                max_value = max(pred1)
                max_index = pred1.index(max_value)
                p = max_index
                if p == label:
                    Accurate += 1
                labelsValues.append(label)

                PredictedValues.append(p)

                counter +=1
                print(pred)

    acc = float(Accurate)/float(counter)
    results = confusion_matrix(labelsValues, PredictedValues)
    print(results)
    print(acc)
    # acc = 100 * float(overallCounter) / float(numIm)

    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt
    emotions = ['angry', 'disgusted', 'feaful','happy', 'neutral', 'sad', 'surprise']
    # lookup = {0: 'fearful', 1: 'angry', 2: 'disgusted', 3: 'neutral', 4: 'surprised', 5: 'happy'}  # , 6:'happy'}
    lookup = {0: "angry", 1: "disgusted", 2:'fearful', 3: "happy", 4: "neutral", 5: 'sad', 6: "surprise"}  # , 6:"reassured"}
    # lookup = {0: 'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
    # lookup = {0: 'Angry', 1: 'Disgust', 2: 'happy', 3: 'neutral', 4: 'surprised', 5: 'Sad', 6: 'fearful'}
    y_true = pd.Series([lookup[_] for _ in labelsValues])  # np.random.random_integers(0, 5, size=100)])
    y_pred = pd.Series([lookup[_] for _ in PredictedValues])  # np.random.random_integers(0, 5, size=100)])

    ###########################################added CONFUSION WITHOU PERCENTAGE
    conf = confusion_matrix(y_true, y_pred)

    conf = confusion_matrix(y_true, y_pred)

    #######################################################################################
    lookup = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: 'neutral', 5: "sad",
              6: "surprise"}  # , 6:"reassured"}

    ####lookup = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "sad", 5: "surprise"}

    # lookup = {0: 'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
    # lookup = {0: 'Angry', 1: 'Disgust', 2: 'happy', 3: 'neutral', 4: 'surprised', 5: 'Sad', 6: 'fearful'}
    y_true = pd.Series([lookup[_] for _ in labelsValues])  # np.random.random_integers(0, 5, size=100)])
    y_pred = pd.Series([lookup[_] for _ in PredictedValues])  # np.random.random_integers(0, 5, size=100)])

    '''print('positive: ' + str(overallCounter))
    print('total: ' + str(numIm))
    print('accuracy: ' + str(acc))'''
    # pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted']).apply(lambda r: r / r.sum(), axis=1)

    res = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], normalize='index')

    ###########################################added CONFUSION WITHOU PERCENTAGE
    conf = confusion_matrix(y_true, y_pred)

    conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]

    # plt.imshow(conf, cmap='Blues', interpolation='nearest')


    # conf = res
    # print(conf)

    plt.imshow(conf, interpolation='nearest', cmap='Blues')
    # plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(emotions))
    plt.xticks(tick_marks, emotions, rotation=45)
    plt.yticks(tick_marks, emotions)

    fmt = '.2f'
    thresh = conf.max() / 2.
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
        plt.text(j, i, format(conf[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    ################################################
    plt.savefig('confusion_matrixCNNNCK.png', format='png')

