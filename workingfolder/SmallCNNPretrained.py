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
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.models import model_from_json
import os
import cv2
from sklearn.metrics import confusion_matrix
# dimensions of our images.
img_width, img_height = 100,100

top_model_weights_path = 'bottleneck_fc_modelCONFCohen1.h5'
train_data_dir = 'dataFromPython3/train'
validation_data_dir = 'dataFromPython3/validation'

test_data_dir = 'dataFromPython3/validation'
nb_train_samples =144#2045 #144 #2045 #2290# 168# 2290 #8 #2318
nb_validation_samples = 35 #689 #689 #35 #689 #770 #41# 770 #1
epochs = 250
batch_size = 1#10

dir = '//Users/emb24/PycharmProjects/emotionfacialunitsvideo1/'

def save_bottlebeck_featuresTest():
    dir = '//Users/emb24/PycharmProjects/emotionfacialunitsvideo1/'
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, 35 // batch_size)
    np.save(open('bottleneck_features_testCONFTESTJAFFE1.npy', 'wb'),
            bottleneck_features_train)



def save_bottlebeck_features():
    dir = '//Users/emb24/PycharmProjects/emotionfacialunitsvideo1/'
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_trainCONFJAFFE1.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, 35 // batch_size)
    np.save(open('bottleneck_features_validationCONFJAFFE1.npy', 'wb'),
            bottleneck_features_validation)



def train_top_model():
    dir = '//Users/emb24/PycharmProjects/emotionfacialunitsvideo1/'
    path = 'dataFromPython3/train/'
    # classify each image in the validation set
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']
    emotionLbls = [[1,0,0,0,0,0],[0,1,0,0,0,0], [0,0,1,0,0,0],[0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]] #,[0,0,0,0,0,0,1]]
    train_data = np.load(open('bottleneck_features_trainCONFJAFFE1.npy', 'rb'))
    labelbyemotion = []
    for emotion in emotions:
        for file in os.listdir(path+emotion):
            if 'png' in file:
                labelbyemotion.append(emotionLbls[emotions.index(emotion)])
    train_labels = np.array(labelbyemotion)

    #[0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validationCONFJAFFE1.npy', 'rb'))
    #validation_labels = np.array(
    #    [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
    path = 'dataFromPython3/validation/'
    labelbyemotionVal = []
    for emotion in emotions:
        for file in os.listdir(path + emotion):
            if 'png' in file:
                labelbyemotionVal.append(emotionLbls[emotions.index(emotion)])
    validation_labels = np.array(labelbyemotionVal)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(1024, activation='relu'))

    #model.add(Dense(1024, activation='relu'))
    #model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

    # serialize model to JSON

    model_json = model.to_json()
    with open( "PretrainCNNCONFJAFFE1.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("PreTraineCNNWeightCONFJAFFE1.h5")
    print("Saved model to disk")


#save_bottlebeck_features()
#train_top_model()


if __name__ == '__main__':

    #save_bottlebeck_features()
    #train_top_model()

    dir = '//Users/emb24/PycharmProjects/emotionfacialunitsvideo1/'
    #classify each image in the validation set
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']

    json_file = open('PretrainCNNCONFJAFFE1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("PreTraineCNNWeightCONFJAFFE1.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    #score = loaded_model.evaluate(X_test, y_test, verbose=0)
    path = 'dataFromPython3/validation/'

    Accurate = 0
    counter = 0
    labelsValues = []
    PredictedValues = []
    save_bottlebeck_featuresTest()

    test_data = np.load(open('bottleneck_features_testCONFTESTJAFFE1.npy', 'rb'))
    counterImage = 0
    for emotion in emotions:
        for file in os.listdir(path+emotion):
            if 'png' in file: # and counterImage< 689:



                image = test_data[counterImage]
                image = image.reshape(1, len(image), len(image),512)
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

                if emotion == 'sad':
                    print(file)
                    print(p)

                counter +=1
                counterImage +=1

                print(counterImage)

    acc = float(Accurate)/float(counter)
    results = confusion_matrix(labelsValues, PredictedValues)
    print(results)
    print(acc)
    # acc = 100 * float(overallCounter) / float(numIm)

    import numpy as np
    import pandas as pd
    import itertools


    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprise']
    # lookup = {0: 'fearful', 1: 'angry', 2: 'disgusted', 3: 'neutral', 4: 'surprised', 5: 'happy'}  # , 6:'happy'}
    lookup = {0: "angry", 1: "disgusted", 2: 'fearful',3: "happy", 4: 'sad', 5: "surprise"}  # , 6:"reassured"}
    # lookup = {0: 'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
    # lookup = {0: 'Angry', 1: 'Disgust', 2: 'happy', 3: 'neutral', 4: 'surprised', 5: 'Sad', 6: 'fearful'}
    y_true = pd.Series([lookup[_] for _ in labelsValues])  # np.random.random_integers(0, 5, size=100)])
    y_pred = pd.Series([lookup[_] for _ in PredictedValues])  # np.random.random_integers(0, 5, size=100)])

    '''print('positive: ' + str(overallCounter))
    print('total: ' + str(numIm))
    print('accuracy: ' + str(acc))'''
    pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted']).apply(lambda r: 100.0 * r / r.sum())

    import matplotlib.pyplot as plt

    conf = confusion_matrix(y_true, y_pred)

    #######################################################################################
    lookup = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4:  "sad",
              5: "surprise"}  # , 6:"reassured"}

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

    y_true1 = np.array(y_true)
    i=0
    y_true1.tofile('y_trueCNNCohenCohen'+str(i)+'.csv', sep=',')
    y_pred1 = np.array(y_pred)
    y_pred1.tofile('y_predCNNCohenCohen'+str(i)+'.csv', sep=',')

    plt.imshow(conf, interpolation='nearest', cmap='Blues')
    # plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(emotions))
    plt.xticks(tick_marks, emotions, rotation=45)
    plt.yticks(tick_marks, emotions)

    fmt = '.3f'
    thresh = conf.max() / 2.
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
        plt.text(j, i, format(conf[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    #######################################################################################

    conf = results
    print(conf)

    norm_conf = []
    for i in conf:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = conf.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    # emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    # emotions = ['angry', 'disgusted', 'happy', 'neutral', 'surprised', 'sad', 'fearful']
    #emotions = ['angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised']



    plt.xticks(range(width), emotions)  # alphabet[:width])
    plt.yticks(range(height), emotions)  # alphabet[:height])
    plt.savefig('confusion_matrixCNNCOHENPreCONF1.png', format='png')

