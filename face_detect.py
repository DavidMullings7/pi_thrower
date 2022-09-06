from genericpath import isdir
from inspect import _void
from types import NoneType
import numpy as np
import cv2
import argparse
import os
import subprocess
import threading

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

soundit = True
parser = argparse.ArgumentParser()
parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. Omit for detecting on default camera.')
parser.add_argument('--image2', '-i2', type=str, help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
parser.add_argument('--database', '-db', type=str, help='Path to the faces database')
parser.add_argument('--image3Text', '-i3t', type=str, help='Name of the input image3.')
parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='face_detection_yunet_2021dec.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv2/opencv2_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv2/opencv2_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
args = parser.parse_args()

def playSound(audio_file):
    subprocess.Popen(["afplay", audio_file])

def recognize(match_face, match_image, recognizer, identities, cosine_similarity_threshold = 0.363):
    """ returns identity with highest cosine similarity index if 
        any identity meets cosine similarity threshold

    Args:
        match_face (_type_): face detected in frame
        match_image (_type_): image where face was detected
        recognizer (_type_): facial recognition model
        identities (_type_): list of identities from database
        cosine_similarity_threshold (float, optional): cosine similarity threshold. Defaults to 0.363.

    Returns:
        _type_: _description_
    """
    similarities: dict[str,list[float]] = {}

    for identity in identities:

        # Align faces
        match_align = recognizer.alignCrop(match_image, match_face)

        # Extract features
        match_features = recognizer.feature(match_align)
        ## [facerecognizer]

        ## [match]
        cosine_score = recognizer.match(match_features, identity[0], cv2.FaceRecognizerSF_FR_COSINE)

        # add cosine score for face to identities matching dictionary
        if identity[1] not in similarities:
            similarities[identity[1]] = []
        similarities[identity[1]].append(cosine_score)

    # get best mean cosine similarity score
    best_identity = max(similarities, key=lambda x: np.mean(similarities[x]))

    # return best matching identity if it exceeds the cosine similarity threshold
    return best_identity if np.mean(similarities[best_identity]) > cosine_similarity_threshold else None


def rect_center(frame: cv2.Mat, coords: np.ndarray) -> bool:
    """ returns True if face is in the center of the frame, False otherwise.

    Args:
        frame (cv2.Mat): frame in which face is located
        coords (np.ndarray): descriptor array of the detected face

    Returns:
        bool
    """    
    frame_center_x: int = frame.shape[1] // 2
    frame_center_y: int = frame.shape[0] // 2
    face_start_x: int = coords[0]
    face_start_y: int = coords[1]
    face_end_x: int = coords[0] + coords[2]
    face_end_y: int = coords[1] + coords[3]
    return face_start_x <= frame_center_x <= face_end_x \
        and face_start_y <= frame_center_y <= face_end_y

def extract_identities(database: str) -> tuple[list[tuple], any]:
    """ uses database on same directory level to return 
        a trained facial recongition model and a list of 
        identities.

    Args:
        database (str): _description_

    Returns:
        tuple[list[tuple], any]: list of database identities and facial recognition model
    """    
    if database is not None:
        identities: list[tuple] = []
        recognizer = cv2.FaceRecognizerSF.create(
                args.face_recognition_model,"")
        for directory in os.listdir(database):
            if  not os.path.isdir('%s/%s' % (database, directory)): continue
            for file in os.listdir('%s/%s' % (database, directory)):
                img = cv2.imread('%s/%s/%s' % (database, directory, file))
                imgWidth = int(img.shape[1]*args.scale)
                imgHeight = int(img.shape[0]*args.scale)

                img = cv2.resize(img, (imgWidth, imgHeight))

                ## [inference]
                # Set input size before inference
                detector.setInputSize((imgWidth, imgHeight))

                faces = detector.detect(img)
                if type(faces[1]) is NoneType: continue

                # Align faces
                face_align = recognizer.alignCrop(img, faces[1][0])

                # Extract features
                face_feature = recognizer.feature(face_align)
                identities.append((face_feature, directory))

        return identities, recognizer
    

def visualize(frame: cv2.Mat, faces, fps, thickness=2, recognizer=None, identities=None):
    """ draws face outlines and identities(if applicable) onto input frame.

    Args:
        frame (cv2.Mat): frame in which face is located.
        faces: list of faces returned from detector.
        fps: frames per second.
        thickness (int, optional): thickness of outline box in pixels.. Defaults to 2.
        recognizer (_type_, optional): facial recognition model. Defaults to None.
        identities (_type_, optional): identities of database individuals. Defaults to None.
    """    
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            # print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv2.circle(frame, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(frame, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(frame, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(frame, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(frame, (coords[12], coords[13]), 2, (0, 255, 255), thickness)

            # print face label if recognized
            if recognizer and identities:
                identity = recognize(face, frame, recognizer, identities)
                if identity and rect_center(frame, coords):
                    cv2.putText(frame, identity, (coords[0] + 15, coords[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == '__main__':

    ## [initialize_FaceDetectorYN]
    detector = cv2.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )

    tm = cv2.TickMeter()

    # If input is an image
    if args.image1 is not None:
        img1 = cv2.imread(cv2.samples.findFile(args.image1))
        img1Width = int(img1.shape[1]*args.scale)
        img1Height = int(img1.shape[0]*args.scale)

        img1 = cv2.resize(img1, (img1Width, img1Height))
        tm.start()

        ## [inference]
        # Set input size before inference
        detector.setInputSize((img1Width, img1Height))

        faces1 = detector.detect(img1)
        ## [inference]

        tm.stop()
        assert faces1[1] is not None, 'Cannot find a face in {}'.format(args.image1)

        # Draw results on the input image
        visualize(img1, faces1, tm.getFPS())

        # Save results if save is true
        if args.save:
            print('Results saved to result.jpg\n')
            cv2.imwrite('result.jpg', img1)

        # Visualize results in a new window
        cv2.imshow("image1", img1)

        if args.image2 is not None:
            img2 = cv2.imread(cv2.samples.findFile(args.image2))

            tm.reset()
            tm.start()
            detector.setInputSize((img2.shape[1], img2.shape[0]))
            faces2 = detector.detect(img2)
            tm.stop()
            assert faces2[1] is not None, 'Cannot find a face in {}'.format(args.image2)
            visualize(img2, faces2, tm.getFPS())
            cv2.imshow("image2", img2)

            ## [initialize_FaceRecognizerSF]
            recognizer = cv2.FaceRecognizerSF.create(
            args.face_recognition_model,"")
            ## [initialize_FaceRecognizerSF]

            ## [facerecognizer]
            # Align faces
            face1_align = recognizer.alignCrop(img1, faces1[1][0])
            face2_align = recognizer.alignCrop(img2, faces2[1][0])

            # Extract features
            face1_feature = recognizer.feature(face1_align)
            face2_feature = recognizer.feature(face2_align)
            ## [facerecognizer]

            cosine_similarity_threshold = 0.363
            l2_similarity_threshold = 1.128

            ## [match]
            cosine_score = recognizer.match(face1_feature, face2_feature, cv2.FaceRecognizerSF_FR_COSINE)
            l2_score = recognizer.match(face1_feature, face2_feature, cv2.FaceRecognizerSF_FR_NORM_L2)
            ## [match]

            msg = 'different identities'
            if cosine_score >= cosine_similarity_threshold:
                msg = 'the same identity'
            print('They have {}. Cosine Similarity: {}, threshold: {} (higher value means higher similarity, max 1.0).'.format(msg, cosine_score, cosine_similarity_threshold))

            msg = 'different identities'
            if l2_score <= l2_similarity_threshold:
                msg = 'the same identity'
            print('They have {}. NormL2 Distance: {}, threshold: {} (lower value means higher similarity, min 0.0).'.format(msg, l2_score, l2_similarity_threshold))
        cv2.waitKey(0)
    else: # Omit input to call default camera

        identities: list[tuple]
        recognizer: any
        identities, recognizer = extract_identities(args.database)

        if args.video is not None:
            print(args.video)
            deviceId = int(args.video)
        else:
            deviceId = 0
        cap = cv2.VideoCapture(deviceId)
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*args.scale)
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*args.scale)
        detector.setInputSize([frameWidth, frameHeight])

        while cv2.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            frame: cv2.Mat = cv2.resize(frame, (frameWidth, frameHeight))

            # Inference
            tm.start()
            faces = detector.detect(frame) # faces is a tuple
            tm.stop()

            # Draw results on the input image
            visualize(frame, faces, tm.getFPS(), recognizer=recognizer, identities=identities)

            # Visualize results
            cv2.imshow('Live', frame)
    cv2.destroyAllWindows()