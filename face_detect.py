from genericpath import isdir
from inspect import _void
from types import NoneType
import numpy as np
import cv2
import argparse
import os
import subprocess
import time
import json

identities: dict[int, str] = None
identity_feature_matrix: np.ndarray = None
clock = time.process_time

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

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

def recognize(match_face, match_image, cosine_similarity_threshold = 0.363):
    """ returns identity with highest cosine similarity index if 
        any identity meets cosine similarity threshold

    Args:
        match_face (_type_): face detected in frame
        match_image (_type_): image where face was detected
        cosine_similarity_threshold (float, optional): cosine similarity threshold. Defaults to 0.363.

    Returns:
        _type_: _description_
    """

    global recognizer
    global identities
    global identity_feature_matrix

    # Align faces
    start = clock()
    match_align = recognizer.alignCrop(match_image, match_face)

    # Extract features
    match_features = np.squeeze(recognizer.feature(match_align))
    end = clock()
    print('Recognize: %.3f' % (end - start))
    
    cosine_similarity_matrix: np.ndarray = np.dot(identity_feature_matrix, match_features) / \
        (np.linalg.norm(identity_feature_matrix, axis=1) * np.linalg.norm(match_features))
    
    argmax_cosine_similarity: np.ndarray = np.argmax(cosine_similarity_matrix)

    if cosine_similarity_matrix[argmax_cosine_similarity] > cosine_similarity_threshold:
        return identities[argmax_cosine_similarity]
    else:
        return None


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

def extract_identities(database: str):
    """ uses database on same directory level to return 
        a trained facial recongition model and a list of 
        identities.

    Args:
        database (str): _description_

    Returns:
        tuple[list[tuple], any]: list of database identities and facial recognition model
    """    

    # import global identity feature matrix
    global identity_feature_matrix
    global identities
    global recognizer

    # read in identity mappings
    with open('%s/identities.json' % (database), 'r') as identities_infile:
        identities = {int(key):value for (key, value) in json.load(identities_infile).items()}

    # read in identity feature matrix
    with open('%s/embeddings.json' % (database), 'r') as embeddings_infile:
        identity_feature_matrix = np.array(json.load(embeddings_infile))

def visualize(frame: cv2.Mat, faces, fps, thickness=2):
    """ draws face outlines and identities(if applicable) onto input frame.

    Args:
        frame (cv2.Mat): frame in which face is located.
        faces: list of faces returned from detector.
        fps: frames per second.
        thickness (int, optional): thickness of outline box in pixels.. Defaults to 2.
    """    


    global recognizer
    global identities

    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv2.circle(frame, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv2.circle(frame, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv2.circle(frame, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv2.circle(frame, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv2.circle(frame, (coords[12], coords[13]), 2, (0, 255, 255), thickness)

            # print face label if recognized
            start = time.process_time()
            if recognizer:
                identity = recognize(face, frame)
                if identity and rect_center(frame, coords):
                    pass
                    cv2.putText(frame, identity, (coords[0] + 15, coords[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            end = time.process_time()
            print('Recognize time: %.3f' % (end - start))

    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (1, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == '__main__':

    global recognizer

    ## [initialize_FaceDetectorYN]
    detector = cv2.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )

    # initialize recognizer
    recognizer = cv2.FaceRecognizerSF.create(
                args.face_recognition_model,"")

    tm = cv2.TickMeter()

    start_1 = time.process_time()
    extract_identities(args.database)
    end_1 = time.process_time()
    print('Extract Identities: %.3f' % (end_1 - start_1))

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
        start = time.process_time()
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        frame: cv2.Mat = cv2.resize(frame, (frameWidth, frameHeight))

        # Inference
        tm.start()
        start = clock()
        faces = detector.detect(frame) # faces is a tuple
        end = clock()
        print('Detect: %.3f' % (end - start))
        tm.stop()

        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        cv2.imshow('Live', frame)
        end = time.process_time()
        print('Frame time: %.3f' % (end - start))
    cv2.destroyAllWindows()