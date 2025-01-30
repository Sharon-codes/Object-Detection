import os
import json
import shutil
from datetime import datetime
import cv2
import numpy as np
import face_recognition
import torch
import torch.nn as nn
from ultralytics import YOLO
import supervision as sv
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import torchvision.transforms as transforms

class PeopleDataManager:
    def __init__(self, data_directory='people_database'):
        self.data_directory = data_directory
        self.images_dir = os.path.join(data_directory, 'images')
        self.profiles_dir = os.path.join(data_directory, 'profiles')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.profiles_dir, exist_ok=True)

    def add_person(self, name, details=None, images=None):
        safe_name = "".join(x for x in name if x.isalnum() or x in [' ', '_']).rstrip().lower().replace(' ', '_')
        person_image_dir = os.path.join(self.images_dir, safe_name)
        os.makedirs(person_image_dir, exist_ok=True)
        profile = {
            'name': name,
            'safe_name': safe_name,
            'registration_date': datetime.now().isoformat(),
            'details': details or {},
            'images': []
        }
        if images:
            for idx, img_path in enumerate(images):
                if not os.path.exists(img_path):
                    print(f"Warning: Image {img_path} does not exist")
                    continue
                file_ext = os.path.splitext(img_path)[1]
                new_image_path = os.path.join(person_image_dir, f'image_{idx}{file_ext}')
                shutil.copy(img_path, new_image_path)
                profile['images'].append(new_image_path)
        profile_path = os.path.join(self.profiles_dir, f'{safe_name}.json')
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=4)
        print(f"Added {name} to database")
        return profile

    def get_known_faces(self):
        known_encodings = []
        known_names = []
        for profile_file in os.listdir(self.profiles_dir):
            if profile_file.endswith('.json'):
                profile_path = os.path.join(self.profiles_dir, profile_file)
                with open(profile_path, 'r') as f:
                    profile = json.load(f)
                for image_path in profile['images']:
                    try:
                        image = face_recognition.load_image_file(image_path)
                        encodings = face_recognition.face_encodings(image)
                        for encoding in encodings:
                            known_encodings.append(encoding)
                            known_names.append(profile['name'])
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
        return known_encodings, known_names

    def add_multiple_people(self, base_directory):
        for person_name in os.listdir(base_directory):
            person_path = os.path.join(base_directory, person_name)
            if os.path.isdir(person_path):
                image_paths = [
                    os.path.join(person_path, img)
                    for img in os.listdir(person_path)
                    if img.lower().endswith(('jpg', 'jpeg', 'png'))
                ]
                if image_paths:
                    self.add_person(
                        name=person_name.replace("_", " "),
                        images=image_paths
                    )

    def rename_person(self, old_name, new_name):
        # Generate safe names for old and new names
        old_safe_name = "".join(x for x in old_name if x.isalnum() or x in [' ', '_']).rstrip().lower().replace(' ', '_')
        new_safe_name = "".join(x for x in new_name if x.isalnum() or x in [' ', '_']).rstrip().lower().replace(' ', '_')
        
        # Paths for old and new profiles
        old_profile_path = os.path.join(self.profiles_dir, f'{old_safe_name}.json')
        new_profile_path = os.path.join(self.profiles_dir, f'{new_safe_name}.json')
        
        # Check if the old profile exists
        if not os.path.exists(old_profile_path):
            print(f"Error: Profile for '{old_name}' does not exist.")
            return
        
        # Update the profile JSON file
        with open(old_profile_path, 'r') as f:
            profile = json.load(f)
        
        profile['name'] = new_name
        profile['safe_name'] = new_safe_name
        
        with open(new_profile_path, 'w') as f:
            json.dump(profile, f, indent=4)
        
        # Rename the directory with images
        old_image_dir = os.path.join(self.images_dir, old_safe_name)
        new_image_dir = os.path.join(self.images_dir, new_safe_name)
        
        if os.path.exists(old_image_dir):
            os.rename(old_image_dir, new_image_dir)
        
        # Remove the old profile JSON file
        os.remove(old_profile_path)
        print(f"Renamed '{old_name}' to '{new_name}' in the database.")


class FaceRecognitionModule:
    def __init__(self, people_data_manager):
        self.people_data_manager = people_data_manager
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        self.known_face_encodings, self.known_face_names = self.people_data_manager.get_known_faces()
        print(f"Loaded {len(self.known_face_names)} known faces")

    def recognize_faces(self, frame):
        try:
            # Resize frame for faster face recognition
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Find all face locations in the image
            face_locations = face_recognition.face_locations(small_frame, model="hog")
            
            if not face_locations:
                return []

            # Compute face encodings
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            
            recognized_faces = []
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                name = "Unknown"
                if len(self.known_face_encodings) > 0:
                    # Compare face with known faces
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                    
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.known_face_names[first_match_index]
                
                recognized_faces.append({
                    'name': name,
                    'location': (top, right, bottom, left)
                })
            
            return recognized_faces

        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
            return []
        
class AdvancedFusionNetwork(nn.Module):
    def __init__(self, input_dim=2048, num_classes=80):
        super(AdvancedFusionNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
        self.classifier = nn.Linear(512, num_classes)
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

class NextGenObjectDetector:
    def __init__(self, object_detection_model='yolov8x.pt', confidence_threshold=0.5, multi_modal_fusion=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Computational Device: {self.device}")
        self.object_detector = YOLO(object_detection_model)
        self.people_data_manager = PeopleDataManager()
        self.face_recognition = FaceRecognitionModule(self.people_data_manager)
        self.confidence_threshold = confidence_threshold
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.transform = transforms.Compose([
            transforms.Resize((640, 480)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect_objects(self, frame):
        if frame is None or frame.size == 0:
            print("Invalid frame")
            return frame

        # Process frame for object detection
        processed_frame = self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)
        results = self.object_detector(frame, conf=self.confidence_threshold)[0]
        detections = sv.Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int)
        )
        
        # Create labels and annotate frame
        labels = [
            f"{results.names[int(class_id)]} {conf:.2f}"
            for class_id, conf in zip(detections.class_id, detections.confidence)
        ]
        
        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )

        # Process person detections for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        person_class_id = 0
        person_indices = np.where(detections.class_id == person_class_id)[0]
        
        for idx in person_indices:
            x1, y1, x2, y2 = map(int, detections.xyxy[idx])
            if y2 > y1 and x2 > x1:
                # Add padding around the person detection
                padding = 50
                y1_pad = max(0, y1 - padding)
                y2_pad = min(frame.shape[0], y2 + padding)
                x1_pad = max(0, x1 - padding)
                x2_pad = min(frame.shape[1], x2 + padding)
                
                person_crop = rgb_frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if person_crop.size > 0:
                    recognized_faces = self.face_recognition.recognize_faces(person_crop)
                    for face in recognized_faces:
                        name = face['name']
                        top, right, bottom, left = face['location']
                        
                        # Adjust coordinates relative to the full frame
                        adjusted_left = x1_pad + left
                        adjusted_top = y1_pad + top
                        
                        # Draw face rectangle
                        cv2.rectangle(
                            annotated_frame,
                            (adjusted_left, adjusted_top),
                            (x1_pad + right, y1_pad + bottom),
                            (0, 255, 0),
                            2
                        )
                        
                        # Add face label with dark background
                        label_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                        cv2.rectangle(
                            annotated_frame,
                            (adjusted_left, adjusted_top - 30),
                            (adjusted_left + label_size[0], adjusted_top - 5),
                            (0, 0, 0),
                            -1
                        )
                        cv2.putText(
                            annotated_frame,
                            name,
                            (adjusted_left, adjusted_top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (255, 255, 255),
                            2
                        )
        
        return annotated_frame

    def real_time_detection(self, save_video=False):
        cap = cv2.VideoCapture(0)
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('detection_output.avi', fourcc, 20.0, (640, 480))
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated_frame = self.detect_objects(frame)
                cv2.imshow('Object and Face Detection', annotated_frame)
                if save_video:
                    out.write(annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            if save_video:
                out.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # Example usage to add multiple people
    people_manager = PeopleDataManager()
    people_manager.rename_person('old_name', 'new_name')
    base_directory = r"C:\Users\sharo\OneDrive\Desktop\Research\Object detection - Research\people_images"  # Replace with your directory path
    people_manager.add_multiple_people(base_directory)

    # Start real-time detection
    detector = NextGenObjectDetector()
    detector.real_time_detection(save_video=True)
