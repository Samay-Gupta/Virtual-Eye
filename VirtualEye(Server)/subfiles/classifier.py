from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imutils import paths
from glob import glob
import numpy as np
import imutils
import pickle
import json
import time
import cv2
import os

#file_syntax = obj_namexDistancexWidth.jpg

class Detector:
    def __init__(self, directory_path):
        self.file_dir = directory_path
        self.label_encoder = None
        self.face_detector = None
        self.data_embedder = None
        self.image_recognizer = None
        self.confidence = 0.5
        self.threshold = 0.3
        self.obj_list = {}
        self.database = {}
        
    def create_dataset(self):
        required_confidence = 0.5
        proto_path = '{}\\dataset_files\\face_data\\deploy.prototxt'.format(self.file_dir)
        model_path = '{}\\dataset_files\\face_data\\res_ssd_300x300.caffemodel'.format(self.file_dir)
        embed_path = '{}\\dataset_files\\face_data\\openface_nn4.small2.v1.t7'.format(self.file_dir)
        output_file = '{}\\dataset_files\\face_data\\embeddings.pickle'.format(self.file_dir)
        image_file_paths = '{}\\images\\face_recognition\\'.format(self.file_dir)
        face_detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        data_embedder = cv2.dnn.readNetFromTorch(embed_path)
        image_files = list(paths.list_images(image_file_paths))
        training_data = []
        training_labels = []
        instances = 0
        for (i, image_path) in enumerate(image_files):
            image_label = image_path.split('\\')[-2]
            image_file = imutils.resize(cv2.imread(image_path), width=600)
            (h, w) = image_file.shape[:2]
            image_blob = cv2.dnn.blobFromImage(
                cv2.resize(
                    image_file, 
                    (300, 300)
                    ), 
                    1.0, 
                    (300, 300),
                    (104.0, 177.0, 123.0), 
                    swapRB=False, 
                    crop=False
                    )
            face_detector.setInput(image_blob)
            image_detections = face_detector.forward()
            if len(image_detections) > 0:
                i = np.argmax(image_detections[0, 0, :, 2])
                confidence = image_detections[0, 0, i, 2]
                if confidence > required_confidence:
                    box_data = image_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x0, y0, x1, y1 = box_data.astype('int')
                    face_image = image_file[y0:y1, x0:x1]
                    if min(list(face_image.shape[:2])) < 20:
                        print(list(face_image.shape[:2]))
                        continue
                    face_image_blob = cv2.dnn.blobFromImage(
                        face_image, 
                        1.0/255, 
                        (96, 96), 
                        (0, 0, 0), 
                        swapRB=True, 
                        crop=False
                        )
                    data_embedder.setInput(face_image_blob)
                    visual_encoder = data_embedder.forward()
                    image_data = visual_encoder.flatten()
                    image_label = image_path.split('\\')[-2]
                    training_data.append(image_data)
                    training_labels.append(image_label)
                    instances += 1
        embedding_data = {
            "data": training_data,
            "labels": training_labels
        }
        with open(output_file, 'wb') as file_object:
            file_object.write(pickle.dumps(embedding_data))

    def load_dataset(self):
        proto_path = '{}\\dataset_files\\face_data\\deploy.prototxt'.format(self.file_dir)
        model_path = '{}\\dataset_files\\face_data\\res_ssd_300x300.caffemodel'.format(self.file_dir)
        embed_path = '{}\\dataset_files\\face_data\\openface_nn4.small2.v1.t7'.format(self.file_dir)
        recog_path = '{}\\dataset_files\\face_data\\recognizer.pickle'.format(self.file_dir)
        output_file = '{}\\dataset_files\\face_data\\embeddings.pickle'.format(self.file_dir)
        encoder_path = '{}\\dataset_files\\face_data\\encoder.pickle'.format(self.file_dir)
        labels_path = '{}\\dataset_files\\object_data\\coco.names'.format(self.file_dir)
        weights_path = '{}\\dataset_files\\object_data\\yolov3.weights'.format(self.file_dir)
        config_path = '{}\\dataset_files\\object_data\\yolov3.cfg'.format(self.file_dir)
        self.face_detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        self.data_embedder = cv2.dnn.readNetFromTorch(embed_path)
        self.image_recognizer = pickle.loads(open(recog_path, 'rb').read())
        self.label_encoder = pickle.loads(open(encoder_path, 'rb').read())
        self.LABELS = open(labels_path).read().strip().split("\n")
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),dtype="uint8")
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        with open('{}\\dataset_files\\database.json'.format(self.file_dir), 'r') as database_file:
            self.database = json.loads(database_file.read())
        self.obj_list = {}
        for r in self.database["synonyms"].keys():
            for x in self.database["synonyms"][r]:
                self.obj_list[x] = r

    def load_image(self, filename):
        return cv2.imread(filename)

    def get_object_characteristics(self, data, name, face=False):
        img_width, img_height = self.database[".image-size"]
        nm, (x, y, w, h) = data
        rel_x, rel_y = (x + w//2), (y + h//3)
        pos_text = "There is {} {}".format('an' if name[0].lower() in ['a', 'e', 'i', 'o', 'u'] else 'a', name)
        if rel_x < img_width*(30/100):
            pos_text += " towards your left"
        elif rel_x <= img_width*(45/100):
            pos_text += " slighly to your left"
        elif rel_x < img_width*(55/100):
            pos_text += " ahead"
        elif rel_x <= img_width*(70/100):
            pos_text += " slightly to your right"
        else:
            pos_text += " towards your right"
        distance = 0
        if face:
            return pos_text
        else:
            if nm  in self.database["distance"].keys() and len(self.database["distance"][nm]) != 0:
                known_width, known_distance = self.database["distance"][nm]
                conv_val = self.database[".inch-to-steps-conv"]
                dist = round(((known_width * known_distance) / w)/conv_val)
                distance = dist
                if dist >= 1:
                    pos_text += " {} steps".format(dist)
            return pos_text, distance

    def person_to_name(self, per_data, face_res):
        cat, (X, Y, W, Y) = per_data
        for (nm , (x, y, w, h)) in face_res:
            midpoint = (x + (w//2), y+(h//2))
            if (midpoint[0] > X and midpoint[0] < X+W) or (midpoint[1] > Y and midpoint[1] < Y+H) and nm != "Unknown":
                per_data[0] = nm
                break
        return per_data

    def compute(self, filename, mode, objs=("!NULL", 0)):
        image_object = self.load_image(filename)
        obj_res_data = self.classify_objects(image_object)
        face_res_data = self.classify_faces(image_object)
        obj_clf_name = [i[0] for i in obj_res_data]
        face_clf_name = [i[0] for i in face_res_data]
        if mode == "FND":
            obj, obj_nm = objs
            if obj in obj_clf_name or obj in face_clf_name:
                obj_data = obj_res_data[obj_clf_name.index(obj)] if obj in obj_clf_name else face_res_data[face_clf_name.index(obj)]
                res = self.get_object_characteristics(obj_data, obj_nm, True)
            else:
                res =  "{} not visible".format(obj)
        else:
            statements = []
            distances = []
            for (i, data) in enumerate(obj_res_data):
                st, dst = self.get_object_characteristics(obj_res_data[i], obj_res_data[i][0], False)
                if data[0].lower() == 'person':
                    obj_res_data[i] = self.person_to_name(data, face_res_data)
                    st = st.replace('a person', obj_res_data[i][0])
                statements.append(st)
                distances.append(dst)
            statements = self.arrange_by_dist(statements, distances)
            if mode == "AST":
                res = statements[0] if len(statements) > 0 else ''
            if mode == "VIS":
                res = ". ".join(statements)
        return res

    def arrange_by_dist(self, st, dst):
        num_val = [i for i in dst if dst[i]!=0]
        if len(num_val) == 0:
            res = st
        else:
            num_val.sort()
            res = []
            while len(num_val) > 0:
                x = dst.index(num_val[0])
                res.append(st[x])
                st[x] = -1
                dst[x] = -1
                del num_val[0]
            rem = [st[i] for i in range(len(dst)) if dst[i]==0]
            res.extend(rem)
        return res
            
    def classify_faces(self, image_object):
        required_confidence = 0.5
        (h, w) = image_object.shape[:2]
        image_blob = cv2.dnn.blobFromImage(
            cv2.resize(
                image_object,
                (300, 300),
                ),
                1.0,
                (300,300),
                (104.0, 117.0, 123.0),
                swapRB=False,
                crop=False
                )
        self.face_detector.setInput(image_blob)
        image_detections = self.face_detector.forward()
        res_data = []
        for i in range(image_detections.shape[2]):
            confidence = image_detections[0, 0, i, 2]
            if confidence > required_confidence:
                box_data = image_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x0, y0, x1, y1 = box_data.astype('int')
                face_image = image_object[y0:y1, x0:x1]
                if min(face_image.shape[:2]) < 20:
                    continue
                face_image_blob = cv2.dnn.blobFromImage(
                    face_image, 
                    1.0/255, 
                    (96, 96), 
                    (0, 0, 0),
                    swapRB=True, 
                    crop=False)
                self.data_embedder.setInput(face_image_blob)
                visual_encoder = self.data_embedder.forward()
                prediction = self.image_recognizer.predict_proba(visual_encoder)[0]
                name = self.label_encoder.classes_[np.argmax(prediction)]
                prediction_data = [name, (x0, y0, x1-x0, y1-y0)]
                res_data.append(prediction_data)
        return res_data

    def classify_objects(self, image_object):
        (H, W) = image_object.shape[:2]
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(image_object, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(ln)
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > self.confidence:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
        res_data = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                res_data.append([self.LABELS[classIDs[i]], (x, y, w, h)])
        return res_data

    def create_distance_dataset(self):
        self.load_dataset()
        img_dir = '{}\\images\\object_detection_distance\\'.format(self.file_dir)
        img_file_list = list(glob(img_dir + '*.jpg'))
        with open('{}\\dataset_files\\database.json'.format(self.file_dir)) as db_file:
            db = json.loads(db_file.read())
        for file_name in img_filoe_list:
            res = (file_name.split('\\')[-1]).split('.')[0]
            nm, D, W = res.split('x')
            nm = nm.replace('_', ' ')
            if nm in db["distance"].keys():
                if len(db["distance"][nm]) == 0:
                    img_obj = self.load_image(file_name)
                    det = self.classify_objects(img_obj)
                    for (name, (X, Y, W, H)) in det:
                        if name == nm:
                            d = (D/W)*w
                            db["distance"][name] = [d, W]
                            break
        with open('{}}\\dataset_files\\database.json'.format(self.file_dir)) as db_file:
            db_file.write(json.dumps(db, indent=4))

    def train_dataset(self):
        self.create_dataset()
        time.sleep(0.5)
        proto_path = '{}\\dataset_files\\face_data\\deploy.prototxt'.format(self.file_dir)
        model_path = '{}\\dataset_files\\face_data\\res_ssd_300x300.caffemodel'.format(self.file_dir)
        embed_path = '{}\\dataset_files\\face_data\\openface_nn4.small2.v1.t7'.format(self.file_dir)
        recog_path = '{}\\dataset_files\\face_data\\recognizer.pickle'.format(self.file_dir)
        output_file = '{}\\dataset_files\\face_data\\embeddings.pickle'.format(self.file_dir)
        encoder_path = '{}\\dataset_files\\face_data\\encoder.pickle'.format(self.file_dir)
        image_recognizer = pickle.loads(open(recog_path, 'rb').read())
        label_encoder = LabelEncoder()
        embed_data = pickle.loads(open(output_file, "rb").read())
        train_data = embed_data['data']
        train_labels = label_encoder.fit_transform(embed_data['labels'])
        clf = SVC(C=1.0, kernel="linear", probability=True)
        clf.fit(train_data, train_labels)
        with open(recog_path, 'wb') as file_object:
            file_object.write(pickle.dumps(clf))
        with open(encoder_path, 'wb') as file_object:
            file_object.write(pickle.dumps(label_encoder))
