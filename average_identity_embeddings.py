import argparse
import cv2
import os
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument('--database', '-db', type=str, help='Path to the faces database', required=True)
parser.add_argument('--face_detection_model', '-fd', type=str, default='face_detection_yunet_2021dec.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv2/opencv2_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv2/opencv2_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
args: argparse.Namespace = parser.parse_args()

if __name__ == '__main__':

    database: str = args.database

    recognizer = cv2.FaceRecognizerSF.create(
            args.face_recognition_model,"")
    detector = cv2.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )

    temp_identities: dict[str, list] = {}
    for directory in os.listdir(database):
        if  not os.path.isdir('%s/%s' % (database, directory)): continue
        temp_identities[directory] = []
        for file in os.listdir('%s/%s' % (database, directory)):
            img = cv2.imread('%s/%s/%s' % (database, directory, file))
            imgWidth = int(img.shape[1]*args.scale)
            imgHeight = int(img.shape[0]*args.scale)

            img = cv2.resize(img, (imgWidth, imgHeight))

            ## [inference]
            # Set input size before inference
            detector.setInputSize((imgWidth, imgHeight))

            faces = detector.detect(img)
            if faces[1] is None: continue

            # Align faces
            face_align = recognizer.alignCrop(img, faces[1][0])

            # Extract features
            face_feature = recognizer.feature(face_align)
            temp_identities[directory].append(face_feature)

    # create identities dictionary which maps indeces to identity names
    identities: dict[int, str] = {i:identity for i, identity in enumerate(temp_identities.keys())}

    # create matrix of average identity embeddigns
    identity_feature_matrix: np.ndarray = np.array([np.squeeze(np.average(np.array(feature), axis=0)) for feature in temp_identities.values()])

    # write identities dictionary to file
    with open('%s/identities.json' % (database), 'w') as identities_outfile:
        json.dump(identities, identities_outfile)

    # write matrix of average identity embeddings to file
    with open('%s/embeddings.json' % (database), 'w') as embeddings_outfile:
        json.dump(identity_feature_matrix.tolist(), embeddings_outfile)