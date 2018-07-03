import os
import cv2
import dlib
import time
import math
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def crop_face_from_frames():
    path = '../video_frames/'
    dataset_path = '../dataset/'
    count = 0
    face_detector = dlib.get_frontal_face_detector()
    # for emotion in emotions:
    #     dataset_path = '../video_frames/' + emotion
    #     frame_path = os.path.join(path, emotion)
    #     # save_path = os.path.join(dataset_path, emotion)
    if 'dataset' not in os.listdir('../'):
        os.mkdir(dataset_path)
    for emotion in emotions:
        # if emotion not in os.listdir(dataset_path):
        #     os.mkdir(dataset_path + emotion)
        frame_path = os.path.join(path, emotion)
        save_path = os.path.join(dataset_path, emotion)
        # Iterate through files
        filelist = [f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))]
        for f in filelist:
            # print(f)
            try:
                # frames = 0
                # vidcap = cv2.VideoCapture('../Trim.mp4')
                # framecount = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
                # framerate = vidcap.get(cv2.CAP_PROP_FPS)
                # while frames < framecount:
                #     start_time = time.time()
                #     _, frame = vidcap.read()
                #     # detect face
                frame = cv2.imread(os.path.join(frame_path, f))
                detected_face = face_detector(frame, 1)
                # crop and save detected face
                if len(detected_face) > 0:
                    for i, d in enumerate(detected_face):
                        crop = frame[d.top():d.bottom(), d.left():d.right()]
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break
                        # save frame as JPEG file
                        cv2.imwrite(save_path + '/0%d.jpg' % count, crop)
                        count += 1
                        # frames += 1
                        # time.sleep(math.floor(framerate/3) / framerate) #extract a frame every 1/3 seconds
            except RuntimeError:
                continue

if __name__ == '__main__':
    crop_face_from_frames()