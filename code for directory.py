import os
import json
import shutil
from datetime import datetime
import cv2
import numpy as np
import face_recognition

class PeopleDataManager:
    def __init__(self, data_directory='people_database'):
        """
        Initialize people data management system
        
        Args:
            data_directory (str): Base directory for storing people data
        """
        self.data_directory = data_directory
        
        # Create main directories
        self.images_dir = os.path.join(data_directory, 'images')
        self.profiles_dir = os.path.join(data_directory, 'profiles')
        
        # Create directories if they don't exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.profiles_dir, exist_ok=True)

    def add_person(self, name, details=None, images=None):
        """
        Add a new person to the database
        
        Args:
            name (str): Full name of the person
            details (dict): Additional personal information
            images (list): List of image paths to add
        """
        # Sanitize name for filename
        safe_name = "".join(x for x in name if x.isalnum() or x in [' ', '_']).rstrip().lower().replace(' ', '_')
        
        # Create person-specific directories
        person_image_dir = os.path.join(self.images_dir, safe_name)
        os.makedirs(person_image_dir, exist_ok=True)
        
        # Prepare profile data
        profile = {
            'name': name,
            'safe_name': safe_name,
            'registration_date': datetime.now().isoformat(),
            'details': details or {},
            'images': []
        }
        
        # Add images
        if images:
            for idx, img_path in enumerate(images):
                # Validate image
                if not os.path.exists(img_path):
                    print(f"Warning: Image {img_path} does not exist")
                    continue
                
                # Copy image to person's directory
                file_ext = os.path.splitext(img_path)[1]
                new_image_path = os.path.join(person_image_dir, f'image_{idx}{file_ext}')
                shutil.copy(img_path, new_image_path)
                
                # Add image to profile
                profile['images'].append(new_image_path)
        
        # Save profile JSON
        profile_path = os.path.join(self.profiles_dir, f'{safe_name}.json')
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=4)
        
        print(f"Added {name} to database")
        return profile

    def get_known_faces(self):
        """
        Retrieve known faces for face recognition
        
        Returns:
            tuple: Lists of face encodings and corresponding names
        """
        known_encodings = []
        known_names = []
        
        # Iterate through profiles
        for profile_file in os.listdir(self.profiles_dir):
            if profile_file.endswith('.json'):
                # Load profile
                profile_path = os.path.join(self.profiles_dir, profile_file)
                with open(profile_path, 'r') as f:
                    profile = json.load(f)
                
                # Process each image
                for image_path in profile['images']:
                    try:
                        # Load image
                        image = face_recognition.load_image_file(image_path)
                        
                        # Compute face encodings
                        encodings = face_recognition.face_encodings(image)
                        
                        # Add valid encodings
                        for encoding in encodings:
                            known_encodings.append(encoding)
                            known_names.append(profile['name'])
                    
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
        
        return known_encodings, known_names

# Modify the FaceRecognitionModule to use PeopleDataManager
class FaceRecognitionModule:
    def __init__(self, people_data_manager):
        """
        Initialize face recognition with people data manager
        
        Args:
            people_data_manager (PeopleDataManager): Manager for people data
        """
        self.people_data_manager = people_data_manager
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Load known faces
        self.load_known_faces()

    def load_known_faces(self):
        """
        Load known faces from people database
        """
        self.known_face_encodings, self.known_face_names = self.people_data_manager.get_known_faces()
        print(f"Loaded {len(self.known_face_names)} known faces")

    def recognize_faces(self, frame):
        """
        Recognize faces in a frame
        
        Args:
            frame (numpy.ndarray): Input image frame
        
        Returns:
            list: Recognized faces with names and locations
        """
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        recognized_faces = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Default name if no match found
            name = "Unknown"
            
            # Compare face to known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding, 
                tolerance=0.6  # Adjust tolerance as needed
            )
            
            # Find best match
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            
            recognized_faces.append({
                'name': name,
                'location': (top, right, bottom, left)
            })
        
        return recognized_faces

# Modify NextGenObjectDetector initialization
class NextGenObjectDetector:
    def __init__(self, 
                 object_detection_model='yolov8x.pt',
                 confidence_threshold=0.5,
                 multi_modal_fusion=True):
        # ... (previous initialization code) ...
        
        # Initialize People Data Manager
        self.people_data_manager = PeopleDataManager('people_database')
        
        # Initialize Face Recognition with People Data Manager
        self.face_recognition = FaceRecognitionModule(self.people_data_manager)

    def add_person(self, name, image_paths, details=None):
        """
        Add a new person to the database
        
        Args:
            name (str): Full name of the person
            image_paths (list): Paths to person's images
            details (dict, optional): Additional person details
        """
        # Add person to database
        profile = self.people_data_manager.add_person(
            name=name,
            images=image_paths,
            details=details
        )
        
        # Reload known faces
        self.face_recognition.load_known_faces()
        
        return profile

# Example Usage
if __name__ == '__main__':
    # Initialize detector
    detector = NextGenObjectDetector()
    
    # Add people to database
    detector.add_person(
        name="John Doe", 
        image_paths=[
            "/path/to/john1.jpg",
            "/path/to/john2.jpg"
        ],
        details={
            'age': 35,
            'occupation': 'Software Engineer'
        }
    )
    
    detector.add_person(
        name="Jane Smith", 
        image_paths=[
            "/path/to/jane1.jpg",
            "/path/to/jane2.jpg"
        ],
        details={
            'age': 28,
            'occupation': 'Data Scientist'
        }
    )
    
    # Start detection
    detector.real_time_detection(save_video=True)