import cv2, glob, random, math, numpy as np, dlib, itertools
from sklearn.svm import SVC
import pickle
import random
#from HeadPoseEstimation import *
from FaceAligner import *
import imutils
from imutils.face_utils import rect_to_bb
#from faceswap import *
from sklearn.model_selection import train_test_split
from scipy import ndimage
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, RandomOverSampler
import collections
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# To calculate a normalized histogram
from scipy.stats import itemfreq
from localbinarypatterns import *


emotions = ['angry', 'disgusted','fearful','happy','sad', 'surprised'] # 'neutral',
#emotions = ['angry', 'disgusted','fearful', 'happy','sad', 'surprised']
emotionslabels = [[1,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0], [0,0,0,0,0,0,1]]

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))


ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)


FACIAL_LANDMARKS_IDXS = collections.OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])



SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")  # Or set this to whatever you named the downloaded file
'''clf = SVC(kernel='linear', probability=True,
          tol=1e-3, class_weight="auto") ''' # , verbose = True) #Set the classifier as a support vector machines with polynomial kernel

#Facial aligner
fa = FaceAligner(predictor, desiredFaceWidth=256) #256
class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass

def get_landmarksFaceSwap(fname):
    im = cv2.imread(fname) #, cv2.IMREAD_COLOR)
    #im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
    #                     im.shape[0] * SCALE_FACTOR))

    im = cv2.resize(im, (100, 100))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #gray = cv2.resize(gray, (150, 150))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)


    frame = gray # cv2.resize(clahe_image, (150, 150)) #clahe_image #cv2.resize(gray, (256, 256))
    rects = detector(frame, 1)

    if len(rects) > 1:
        #raise TooManyFaces
        s1 = 'error'
        print(s1)
    elif len(rects) == 0:
        #raise NoFaces
        s1 = 'error'
        print(s1)
    else:
        land =  np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
        s1 = transformation_from_points(land) #[ALIGN_POINTS])
    return s1

def classifaction_report_csv(report, pathFile):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-2]:
        if len(line) > 0:
            row = {}
            line = str(line)
            line =line.replace(' ', 'L')
            row_data = line.split('L')
            while '' in row_data:
                row_data.remove('')
            row['class'] = row_data[0]
            row['precision'] = row_data[1]
            row['recall'] = row_data[2]
            row['f1_score'] = row_data[3]
            row['support'] = row_data[4]


            report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(pathFile, index = False)

def create_class_weight(labels_dict, maxCount):
    from sklearn.utils import class_weight
    total = np.sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weightdic = dict()


    class_weight1 = class_weight.compute_class_weight('balanced', np.unique(npar_trainlabs), npar_trainlabs)

    for key in keys:
        score =  total/float(labels_dict[key])
        if (labels_dict[key] < maxCount/2):

            score = (maxCount/float(labels_dict[key]))* maxCount
        else:
            score = (maxCount / float(labels_dict[key]))
        #score = class_weight1[key] #math.log(mu*total/float(labels_dict[key]))

        #score = total/float(labels_dict[key])
        class_weightdic[key] = score #if score > total/float(max) else 1.0 #1.0 else 1.0



    return class_weightdic

def unique_rows_counts(a):

    # Calculate linear indices using rows from a
    lidx = np.ravel_multi_index(a.T,a.max(0)+1 )

    # Get the unique indices and their counts
    _, unq_idx, counts = np.unique(lidx, return_index = True, return_counts=True)

    # return the unique groups from a and their respective counts
    return a[unq_idx], counts


def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    path = 'NewAllImages/' #'NewAllImages/' #'NewAllImages/' #sorted_set/' #Face/' #'NewAllImages/' #'NewAllImages/' #sorted_set/' #Face/' #sorted_set/'
    #pathJaffe = "//Users/emb24/Documents/PHd2/PhD/PycharmProjects/PivotHeadTest/jaffe/"
    #path = pathJaffe
    dir=path+"%s/*" % emotion
    files = glob.glob(dir)
    print(emotion+':'+str(len(files)))
    random.shuffle(files)
    training = files[:int(len(files) * .20)]  # get first 80% of file list
    prediction = files[-int(len(files) * .80):]  # get last 20% of file list
    testing =  prediction #files[-int(len(files) * 0.20):] ####prediction

    return training, prediction, testing

def transformation_from_points(points1):


    points1 = points1.astype(np.float64)


    c1 = np.mean(points1, axis=0)

    points1 -= c1


    s1 = np.std(points1)

    points1 /= s1


    #return the standarisation of the points


    return points1



def get_landmarksDIST(pic):
    landmarks_vectorised1 = []
    #detector = dlib.get_frontal_face_detector()
    frame = cv2.imread(pic)

    frame = cv2.resize(frame, (300, 300))

    #frame = cv2.resize(frame, (256, 256))

    #image = imutils.resize(frame, width=40)  # , length = 40)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    '''equ = cv2.equalizeHist(gray)
    res = np.hstack((gray, equ))  # stacking images side-by-side'''



    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    clahe_image = clahe.apply(gray)
    ####
    clahe_image = gray
    ####
    rects = detector(clahe_image, 1)

    # loop over the face detections
    #print(pic)
    #print("Number of faces detected: {}".format(len(rects)))
    winSize = (64, 64)
    blockSize = (16, 16) #16
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    #hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
    #                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    hog = cv2.HOGDescriptor()
    ########
    desc =  LocalBinaryPatterns(250, 8)
    radius =16
    # Number of points to be considered as neighbourers
    no_points = 16 #8 * radius
    # Uniform LBP is used

    counter = 0
    #########
    for rect in rects:
        face = 1
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        d = rect

        img = clahe_image  # np.float32(frame) / 255.0;
        crop = img[d.top():d.bottom(), d.left():d.right()]

        faceAligned = clahe_image # fa.align(frame, gray, rect)

        #faceAligned = fa.align(frame, gray, rect)

        #faceAligned = fa.align(frame, gray, rect)

        winStride = (8, 8)
        padding = (8, 8)
        locations = ((10, 20),)
    #detections = rects #detector(image, 1)
    #d = rect
    #test =0
    #if test == 0: #for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(clahe_image, rect)  # Draw Facial Landmarks with the predictor class
        #landmarksPoints = np.matrix([[p.x, p.y] for p in predictor(image, detections[0]).parts()])
        landmarksPoints = np.matrix([[p.x, p.y] for p in predictor(faceAligned, rect).parts()])
        ##Get centre of mass

        #####hog features scickit
        from skimage.feature import hog as hog1
        from skimage import data, color, exposure

        #crop = cv2.resize(crop, (150, 150))

        '''hog_image = hog1(faceAligned, orientations=8, pixels_per_cell=(32, 32), # 32
                            cells_per_block=(1, 1)) #, visualize=True)'''

        features, hog_image = hog1(clahe_image, orientations=8, pixels_per_cell=(32,32),
                                  cells_per_block=(1, 1), visualise=True)

        #fd, hog_image = hog(image, orientations=8, pixels_per_cell=(32, 32),
        #                    cells_per_block=(1, 1), visualize=True)
        hog_features = features
        hog_images = hog_image
        #h = hog.compute(faceAligned, winStride, padding, locations)
        hog_image = hog_image # np.float32(hog_features) #hog_image)
        h = hog_image
        h = h.flatten()

        image = clahe_image

        '''fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        ax1.set_adjustable('box-forced')

        # Rescale histogram for better display
        #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        hog_image_rescaled = hog_image
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        ax1.set_adjustable('box-forced')
        plt.show()'''


        ######get local binary pattern

        #histLBP = desc.describe(crop)

        # Calculate the histogram

        lbp = local_binary_pattern(crop, no_points, radius, method='uniform') #''default')
        lbparray = np.array(lbp)
        import scipy
        histogram = scipy.stats.itemfreq(lbp)


        lbparray =  lbparray.flatten() #histogram.flatten()
        x = itemfreq(lbp.ravel())
        # Normalize the histogram
        histLBP =lbparray # x[:, 1] / sum(x[:, 1])
        # Append image path in X_name

        #######################

        xlist = []
        ylist = []
        allcoords = []

        for i in range(1, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            coords = [float(shape.part(i).x),float(shape.part(i).y)]
            allcoords.append(coords)



        ###### TODO write ANIMA inspired code here
        # get centre of left brows
        # get centre of left eye
        #distance between brows and eyes centres (add to features array)
        #get centre right brows
        #get centre right eyes
        #distance between brows and eyes centres (add this to feature array )
        #distance between centre left brows and right brows
        #get centre of upper lip
        #get centre of lower lips
        #get distance between centres upper and lower lips


        #################################

        xmean = np.mean(xlist)  # Get the mean of both axes to determine centre of gravity
        ymean = np.mean(ylist)
        xcentral =[(x - xmean) for x in xlist]  # get distance between each point and the central point in both axes
        ycentral = [(y - ymean) for y in ylist]

        if xlist[26] == xlist[
            29]:  # If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
            anglenose = 0
        else:
            anglenose = int(math.atan((ylist[26] - ylist[29]) / (xlist[26] - xlist[29])) * 180 / math.pi)

        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90

        landmarks_vectorised = []

        centreMass = ndimage.measurements.center_of_mass(np.array(allcoords))
        '''p1, p2 = HeadPoseEstimation(faceAligned, landmarksPoints)
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)'''
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((w, z))
            #TODO : DISTANCE FROM CENTRE GRAVITY TO EACH POINT

            #DistCentreMass = np.linalg.norm(coornp - centreMass)
            DistCentreMass = np.linalg.norm(coornp - meannp)
            #add points for head pose and distance for each point



            '''distp1 = np.linalg.norm(coornp - p1)
            distp1X = p1[0]
            distp1Y = p1[1]
            distp2 = np.linalg.norm(coornp - p2)
            distp2X = p2[0]
            distp2Y = p2[1]'''
            #sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            dist = np.linalg.norm(coornp - meannp)
            anglerelative = (math.atan((z - ymean) / (w - xmean)) * 180 / math.pi) - anglenose
            #landmarks_vectorised.append(dist)
            '''landmarks_vectorised.append(distp1)
            landmarks_vectorised.append(distp2)'''

            '''landmarks_vectorised.append(distp1X)
            landmarks_vectorised.append(distp1Y)

            landmarks_vectorised.append(distp2X)
            landmarks_vectorised.append(distp2Y)'''

            #landmarks_vectorised.append(DistCentreMass)

            #landmarks_vectorised.append(dist)
            #landmarks_vectorised.append(anglerelative)



            '''cv2.circle(frame, (int(coornp[0]), int(coornp[1])), 1, (255, 0, 0), -1)

            p1 = (int(coornp[0]), int(coornp[1]))
            #p2 = (int(centreMass[0]), int(centreMass[1]))
            p2 = (int(xmean), int(ymean))
            if counter < 2:
                cv2.line(frame, p1, p2, (255,0, 0), 1)
            counter +=1'''
    ###cv2.imwrite('CentreMassCK.png', frame)
    if len(rects) < 1:
        landmarks_vectorised2 = "error"
    else:

        #landmarks_vectorised2 = np.array(landmarks_vectorised)
        landmarks_vectorised2 = h # np.array(h) #np.concatenate((landmarks_vectorised2, h), axis=0)
        #landmarks_vectorised2 = np.array(histLBP)
        #landmarks_vectorised2 =  np.concatenate((landmarks_vectorised2, h), axis=0)
    #landmarks_vectorised2 = landmarks_vectorised2.reshape(67,6,1)
    #landmarks_vectorised1.append(landmarks_vectorised2)
    #landmarks_vectorised1 = np.array(landmarks_vectorised1)
    #landmarks_vectorised1 = np.array(landmarks_vectorised2)
    return landmarks_vectorised2



def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    testing_data = []
    testing_labels = []
    trainingF = []
    predictionF = []
    testingF = []
    for emotion in emotions:
        print(emotion)
        training, prediction,testing = get_files(emotion)
        trainingF += training
        predictionF += prediction
        testingF += testing
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            '''image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = get_landmarks(clahe_image)'''
            landmarks_vectorised1 = get_landmarksDIST(item)
            '''landmarks_vectorised1 = get_landmarksFaceSwap(item)  # get_landmarksDIST(item)
            ## only for standard deviation and centroid method
            landmarks_vectorised1 = np.array(landmarks_vectorised1)
            landmarks_vectorised1 = landmarks_vectorised1.flatten()'''
            landmarks_vectorised =   landmarks_vectorised1 #np.concatenate((landmarks_vectorised1, landmarks_vectorised12), axis=0) #landmarks_vectorised1 #np.concatenate((landmarks_vectorised2, landmarks_vectorised1), axis = 0)

            if landmarks_vectorised1 == "error": # or landmarks_vectorised1 == "error" :
             pass
            else:
                training_data.append(landmarks_vectorised)  # append image array to training data list
                training_labels.append(emotions.index(emotion)) #emotionslabels[emotions.index(emotion)])

        for item in prediction:
            '''image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = get_landmarks(clahe_image)'''
            landmarks_vectorised1 = get_landmarksDIST(item)
            '''landmarks_vectorised1 = get_landmarksFaceSwap(item)  # get_landmarksDIST(item)
            ## only for standard deviation and centroid method
            landmarks_vectorised1 = np.array(landmarks_vectorised1)
            landmarks_vectorised1 = landmarks_vectorised1.flatten()'''
            landmarks_vectorised = landmarks_vectorised1 # np.concatenate((landmarks_vectorised1, landmarks_vectorised12), axis=0) #landmarks_vectorised1 # np.concatenate((landmarks_vectorised2, landmarks_vectorised1), axis=0)
            if landmarks_vectorised1 == "error":
                pass
            else:
                prediction_data.append(landmarks_vectorised)
                prediction_labels.append(emotions.index(emotion)) #emotionslabels[emotions.index(emotion)]) #emotions.index(emotion))

        for item in testing:
            '''image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = get_landmarks(clahe_image)'''
            landmarks_vectorised1 = get_landmarksDIST(item)
            '''landmarks_vectorised1 = get_landmarksFaceSwap(item)  # get_landmarksDIST(item)
            ## only for standard deviation and centroid method
            landmarks_vectorised1 = np.array(landmarks_vectorised1)
            landmarks_vectorised1 = landmarks_vectorised1.flatten()'''
            landmarks_vectorised =  landmarks_vectorised1 ##np.concatenate((landmarks_vectorised1, landmarks_vectorised12), axis=0)
            if landmarks_vectorised1 == "error":
                pass
            else:
                testing_data.append(landmarks_vectorised)
                testing_labels.append(emotions.index(emotion)) #emotionslabels[emotions.index(emotion)]) #emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels, testing_data, testing_labels,trainingF, predictionF,testingF


accur_lin = []


if __name__ == '__main__':

    pathMain = '//Users/emb24/PycharmProjects/EmotionFacialUnitsVideo/'
    ind = 0
    for i in range(0, 1):
        ind +=1

        print("Making sets %s" % i)  # Make sets by random sampling 80/20%
        training_data, training_labels, prediction_data, prediction_labels, testing_data, testing_labels,training, prediction,testing = make_sets()

        npar_train = np.array(training_data)  # Turn the training set into a numpy array for the classifier
        npar_train = training_data

        npar_trainlabs = np.array(training_labels)

        npar_test = np.array(testing_data)  # Turn the training set into a numpy array for the classifier
        npar_test = testing_data

        npar_testlabs = np.array(testing_labels)




        from sklearn import preprocessing


        #class weight

        #unqrows, counts = unique_rows_counts(npar_trainlabs)

        '''u,counts = np.unique(npar_trainlabs, return_counts = True)


        maxCount = np.max(counts)
        labeldics = {}
        labeldics = {0: counts[0], 1: counts[1], 2: counts[2], 3: counts[3], 4: counts[4], 5: counts[5]}

        weights = create_class_weight(labeldics, maxCount)'''




        clf = SVC(kernel='linear', probability=True,
                 tol=1e-3 ,
                 class_weight='balanced') #{0:.09, 1:.1, 2:.01, 3:.00005,4:.005,5:.05})  # , verbose = True) #Set the classifier as a support vector machines with polynomial kernel




        #npar_train = preprocessing.scale(npar_train, axis = 0)

        print("training SVM linear %s" % i)  # train SVM

        ####over samplling TEST

        filename = 'MODELCohenSVMHOGCONF80201JAFFE' + str(i) + '.sav'

        clf = pickle.load(open(filename, 'rb'))

        #X,y = make_classification(n_classes=2, weights = [0.1,0.9], n_features = 20, n_samples = 5000)
        RND_SEED = 0
        sm = RandomOverSampler(ratio = 'minority', random_state=RND_SEED)
        #npar_train, npar_trainlabs = sm.fit_sample(npar_train,npar_trainlabs)
        #########

        #clf.fit(npar_train, npar_trainlabs) #,class_weight={0:.2, 1:.2, 2:.1, 3:.05,4:.5,5:.05}) #,class_weight='balanced') #training_labels)

        print("getting accuracies %s" % i)  # Use score() function to get accuracy

        npar_pred = np.array(prediction_data)
        npar_predLabels = np.array(prediction_labels)

        #npar_pred = preprocessing.scale(npar_pred, axis = 0)

        #npar_pred, npar_predLabels = sm.fit_sample(npar_pred, npar_predLabels)
        ###pred_lin = clf.score(npar_pred, npar_predLabels)
        ##print( "Accuracy: ", pred_lin)
        ###accur_lin.append(pred_lin)  # Store accuracy in a list

        '''filename = '//Users/emb24/PycharmProjects/RNN-LSTMVideo/' + 'MODELNEw1CohenSVMCoordCohen8050STSortedSetNoneutral'+str(i)+ '.sav'
        filename =  'MODELNEw1CohenSVMCoordCohen8050LBPNewImages' + str(i) + '.sav'
        pickle.dump(clf, open(filename, 'wb'))

        filename =  'MODELNEw1CohenSVMCoordCohen8050SortedSetNoneutral' + str(i) + '.sav''
        clf = pickle.load(open(filename, 'rb'))'''

        '''filename = 'MODELCohenSVMHOGCONF' + str(i) + '.sav'

        clf = pickle.load(open(filename, 'rb'))'''



        #pickle.dump(clf, open(filename, 'wb'), protocol=2)

        counter = 0
        labelsValues = []
        AccuracyCounter = 0
        TotalCounter = 1
        PredictedValues = []
        npar_test = npar_test #npar_pred
        npar_testlabs = npar_testlabs # npar_predLabels
        folderresults = 'AllResultsValidation'
        #ProbaFile = open(pathMain + folderresults+'/' + 'FileProbaCoord'+str(ind)+'.txt', 'w')
        ProbaFile = open(folderresults + '/' + 'FileProbaresultsCKCoord' + str(ind) + '.txt', 'w')
        a = 0
        for emotion in range(0,1): #emotions:
                for testData in npar_test:

                        if a == 0: #counter < len(npar_testlabs):
                            testData = testData.reshape(1,len(testData))
                            node = clf.predict_proba(testData)
                            node_id = clf.predict(testData)
                            em = npar_testlabs[counter] #semotions.index(emotion)



                            '''file = testing[counter] #prediction[counter]

                            ProbaFile.write("%s\n" % file)
                            ProbaFile.write("%s\n" % node)'''




                            labelsValues.append(em)
                            PredictedValues.append(int(node_id[0]))
                            if em == int(node_id[0]):
                                AccuracyCounter +=1

                            TotalCounter +=1
                            counter +=1






        t = accuracy_score(labelsValues, PredictedValues)
        print('accuracy'+str(t))
        print('Test Acc:'+ str(float(AccuracyCounter)/float(TotalCounter)))

        results = confusion_matrix(labelsValues, PredictedValues)
        print(results)
        #acc = 100 * float(overallCounter) / float(numIm)

        import numpy as np
        import pandas as pd


        #emotions = ['angry', 'disgusted', 'fearful','happy','neutral' , 'sad', 'surprise'] #'neutral',

        emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']
        #emotions = ['angry', 'disgusted', 'fearful,''happy', 'sad', 'surprise']

        #lookup = {0: 'fearful', 1: 'angry', 2: 'disgusted', 3: 'neutral', 4: 'surprised', 5: 'happy'}  # , 6:'happy'}

        lookup = {0:"angry", 1: "disgusted", 2:"fearful", 3:"happy", 4:"sad", 5:"surprise"} #, 6:"reassured"}

        ####lookup = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "sad", 5: "surprise"}

        #lookup = {0: 'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
        #lookup = {0: 'Angry', 1: 'Disgust', 2: 'happy', 3: 'neutral', 4: 'surprised', 5: 'Sad', 6: 'fearful'}
        y_true = pd.Series([lookup[_] for _ in labelsValues])  # np.random.random_integers(0, 5, size=100)])
        y_pred = pd.Series([lookup[_] for _ in PredictedValues])  # np.random.random_integers(0, 5, size=100)])

        '''print('positive: ' + str(overallCounter))
        print('total: ' + str(numIm))
        print('accuracy: ' + str(acc))'''
        #pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted']).apply(lambda r: r / r.sum(), axis=1)

        res = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], normalize='index')





        ###########################################added CONFUSION WITHOU PERCENTAGE
        conf = confusion_matrix(y_true, y_pred)

        conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
	#y_true1 = np.array(y_true)
	#y_true1.tofile('y_true.csv', sep=',')
	#y_pred1 =np.array(y_pred)
	#y_pred1.tofile('y_pred.csv',sep=',')
	#plt.imshow(conf, cmap='Blues', interpolation='nearest')
	#y_true1= np.array(y_true)
	#y_true1.tofile('y_true.csv', sep=',')
	#y_pred1 = np.array(y_pred)
	#y_pred1.tofile('y_pred.csv',sep=',')
        #print(conf)
        y_true1 = np.array(y_true)
        y_true1.tofile('y_trueCohenJAFFE'+str(i)+'.csv', sep=',')
        y_pred1 = np.array(y_pred)
        y_pred1.tofile('y_predCohenJAFFE'+str(i)+'.csv', sep=',')
        plt.imshow(conf, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
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
        #plt.savefig(folderresults+'/'+'confusion_matrixresultsCKCoord8020'+str(ind)+'.png', format='png')


        ################################################

        '''norm_conf = []
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
        #emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

        #emotions = ['angry', 'disgusted', 'happy', 'neutral', 'surprised', 'sad', 'fearful']

        #emotions = ['angry', 'disgusted', 'fearful', 'happy','sad', 'surprised']


        plt.xticks(range(width), emotions)  # alphabet[:width])
        plt.yticks(range(height), emotions)  # alphabet[:height])
        '''
        print(classification_report(y_true, y_pred, target_names=emotions))
        results = classification_report(y_true, y_pred, target_names=emotions)
        classifaction_report_csv(results, folderresults+'/'+'resultsCKCoord'+str(ind)+'.csv')

        #plt.savefig(folderresults+'/'+'confusion_matrixresultsCKCoord8020'+str(ind)+'.png', format='png')

thefile = open(folderresults+'/'+'accuraciesresultsCKCoord.txt', 'w')
for item in accur_lin:
  thefile.write("%s\n" % item)
thefile.write("%s\n" % np.mean(accur_lin))
print("Mean value lin svm: %.3f" % np.mean(accur_lin))  # Get mean accuracy of the 10 runs
