import os
import json
import shutil
from datetime import datetime
import cv2
import numpy as np
import face_recognition
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading
import queue
import speech_recognition as sr
import pyttsx3
import pandas as pd
import logging
import pyaudio
from test1 import PeopleDataManager, NextGenObjectDetector

def migrate_database_structure():
    """
    Migrate the existing nested structure to the correct format
    """
    database_path = 'people_database'
    images_dir = os.path.join(database_path, 'images')
    profiles_dir = os.path.join(database_path, 'profiles')
    
    print("\nMigrating database structure...")
    
    # First, load profile data
    profiles = {}
    if os.path.exists(profiles_dir):
        for profile_file in os.listdir(profiles_dir):
            if profile_file.endswith('.json'):
                with open(os.path.join(profiles_dir, profile_file), 'r') as f:
                    person_name = profile_file.replace('.json', '')
                    profiles[person_name] = json.load(f)
    
    # Migrate images from nested structure
    if os.path.exists(images_dir):
        for person_folder in os.listdir(images_dir):
            nested_path = os.path.join(images_dir, person_folder)
            if os.path.isdir(nested_path):
                # Create new person directory in main database folder
                new_person_path = os.path.join(database_path, person_folder)
                if not os.path.exists(new_person_path):
                    os.makedirs(new_person_path)
                    print(f"\nCreated directory: {new_person_path}")
                
                # Move images
                for img in os.listdir(nested_path):
                    if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                        old_path = os.path.join(nested_path, img)
                        new_path = os.path.join(new_person_path, img)
                        if not os.path.exists(new_path):
                            shutil.copy2(old_path, new_path)
                            print(f"Copied {img} to {new_person_path}")
                
                # Save profile data if exists
                if person_folder in profiles:
                    profile_path = os.path.join(new_person_path, 'profile.json')
                    with open(profile_path, 'w') as f:
                        json.dump(profiles[person_folder], f, indent=4)
                    print(f"Saved profile data for {person_folder}")
    
    print("\nMigration complete. Verifying new structure...")
    list_directory_contents()
    
    # Don't delete old directories yet - keep as backup
    print("\nNote: Original 'images' and 'profiles' directories have been kept as backup.")
    print("Once you verify everything is correct, you can delete them manually.")

def list_directory_contents():
    """
    List all contents of the people_database directory
    """
    database_path = 'people_database'
    print("\nCurrent database directory structure:")
    for root, dirs, files in os.walk(database_path):
        level = root.replace(database_path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

def verify_face_images():
    """
    Verify that face images can be loaded and processed
    """
    database_path = 'people_database'
    print("\nVerifying face images:")
    
    for person_folder in os.listdir(database_path):
        person_path = os.path.join(database_path, person_folder)
        if os.path.isdir(person_path) and not person_folder in ['images', 'profiles']:
            images = [f for f in os.listdir(person_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if images:
                print(f"\nChecking images for {person_folder}:")
                for img in images:
                    try:
                        image_path = os.path.join(person_path, img)
                        face_image = face_recognition.load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(face_image)
                        
                        if face_encodings:
                            print(f"  ✓ {img}: Found {len(face_encodings)} face(s)")
                        else:
                            print(f"  ✗ {img}: No faces detected")
                    except Exception as e:
                        print(f"  ✗ {img}: Error processing image: {str(e)}")
            else:
                print(f"\n✗ No images found for {person_folder}")
class AITutor:
    def __init__(self):
        self.audio_available = False
        try:
            
            self.recognizer = sr.Recognizer()
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            self.audio_available = True
            print("Audio system initialized successfully")
        except ImportError:
            print("\nPyAudio is not installed. Using text-only mode...")
            
        try:
            self.model_name = "tiiuae/falcon-7b-instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Modified to use CPU instead of CUDA
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Changed from bfloat16 to float32
                trust_remote_code=True,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self.device = torch.device('cpu')
            self.model = self.model.to(self.device)
            print("AI model initialized successfully on CPU")
        except Exception as e:
            print(f"Error initializing AI model: {str(e)}")
            self.model = None
            self.tokenizer = None

    def generate_response(self, question, learning_style):
        """Generate a response using the AI model based on learning style"""
        if not self.model or not self.tokenizer:
            return "I apologize, but the AI model is not properly initialized."

        try:
            # Create specialized prompts based on learning style
            if learning_style == "real_life":
                prompt = (
                    f"Question: {question}\n"
                    "Instruction: Explain this concept using real-life examples that "
                    "anyone can relate to. Include at least 2 practical examples from "
                    "everyday life. Make it engaging and connected to daily experiences."
                )
            elif learning_style == "theoretical":
                prompt = (
                    f"Question: {question}\n"
                    "Instruction: Provide a theoretical explanation in simple terms. "
                    "Break down complex concepts into understandable parts. Use clear "
                    "language while maintaining academic accuracy. Avoid jargon unless "
                    "necessary, and when used, explain it simply."
                )
            else:  # simplified
                prompt = (
                    f"Question: {question}\n"
                    "Instruction: Explain this in the simplest possible terms. Use "
                    "multiple easy-to-understand examples. Break everything down into "
                    "basic concepts. Use analogies where helpful. Avoid any complex "
                    "terminology. Provide at least 3 simple examples to illustrate "
                    "the concept."
                )

            # Generate response with the model using CPU
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():  # Added to reduce memory usage
                outputs = self.model.generate(
                    **inputs,
                    max_length=800,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the actual response part (after the instruction)
            response = response.split("Response:")[-1].strip()
            
            # Post-process the response
            if learning_style == "real_life":
                response = "Let me explain this with some real-life examples:\n\n" + response
            elif learning_style == "theoretical":
                response = "Here's a simple theoretical explanation:\n\n" + response
            else:
                response = "Let me break this down in the simplest way possible with examples:\n\n" + response
                
            return response
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return self.get_fallback_response(question)
        

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler and a stream handler
file_handler = logging.FileHandler('app.log')
stream_handler = logging.StreamHandler()

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Replace print statements with logger statements
def migrate_database_structure():
    """
    Migrate the existing nested structure to the correct format
    """
    database_path = 'people_database'
    images_dir = os.path.join(database_path, 'images')
    profiles_dir = os.path.join(database_path, 'profiles')
    
    logger.info("\nMigrating database structure...")
    
    # First, load profile data
    profiles = {}
    if os.path.exists(profiles_dir):
        for profile_file in os.listdir(profiles_dir):
            if profile_file.endswith('.json'):
                with open(os.path.join(profiles_dir, profile_file), 'r') as f:
                    person_name = profile_file.replace('.json', '')
                    profiles[person_name] = json.load(f)
    
    # Migrate images from nested structure
    if os.path.exists(images_dir):
        for person_folder in os.listdir(images_dir):
            nested_path = os.path.join(images_dir, person_folder)
            if os.path.isdir(nested_path):
                # Create new person directory in main database folder
                new_person_path = os.path.join(database_path, person_folder)
                if not os.path.exists(new_person_path):
                    os.makedirs(new_person_path)
                    logger.info(f"\nCreated directory: {new_person_path}")
                
                # Move images
                for img in os.listdir(nested_path):
                    if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                        old_path = os.path.join(nested_path, img)
                        new_path = os.path.join(new_person_path, img)
                        if not os.path.exists(new_path):
                            shutil.copy2(old_path, new_path)
                            logger.info(f"Copied {img} to {new_person_path}")
                
                # Save profile data if exists
                if person_folder in profiles:
                    profile_path = os.path.join(new_person_path, 'profile.json')
                    with open(profile_path, 'w') as f:
                        json.dump(profiles[person_folder], f, indent=4)
                    logger.info(f"Saved profile data for {person_folder}")
    
    logger.info("\nMigration complete. Verifying new structure...")
    list_directory_contents()
    
    # Don't delete old directories yet - keep as backup
    logger.info("\nNote: Original 'images' and 'profiles' directories have been kept as backup.")
    logger.info("Once you verify everything is correct, you can delete them manually.")

def list_directory_contents():
    """
    List all contents of the people_database directory
    """
    database_path = 'people_database'
    logger.info("\nCurrent database directory structure:")
    for root, dirs, files in os.walk(database_path):
        level = root.replace(database_path, '').count(os.sep)
        indent = ' ' * 4 * level
        logger.info(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            logger.info(f"{subindent}{f}")

def verify_face_images():
    """
    Verify that face images can be loaded and processed
    """
    database_path = 'people_database'
    logger.info("\nVerifying face images:")
    
    for person_folder in os.listdir(database_path):
        person_path = os.path.join(database_path, person_folder)
        if os.path.isdir(person_path) and not person_folder in ['images', 'profiles']:
            images = [f for f in os.listdir(person_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if images:
                logger.info(f"\nChecking images for {person_folder}:")
                for img in images:
                    try:
                        image_path = os.path.join(person_path, img)
                        face_image = face_recognition.load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(face_image)
                        
                        if face_encodings:
                            logger.info(f"  {img}: Found {len(face_encodings)} face(s)")
                        else:
                            logger.info(f"  {img}: No faces detected")
                    except Exception as e:
                        logger.info(f"  {img}: Error processing image: {str(e)}")
            else:
                logger.info(f"\n No images found for {person_folder}")

class AITutor:
    def __init__(self):
        self.audio_available = False
        try:
            
            self.recognizer = sr.Recognizer()
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
            self.audio_available = True
            print("Audio system initialized successfully")
        except ImportError:
            print("\nPyAudio is not installed. Using text-only mode...")
            
        try:
            # Updated model initialization without trust_remote_code
            print("Initializing AI model...")
            self.model_name = "tiiuae/falcon-7b-instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                low_cpu_mem_usage=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("AI model initialized successfully")
        except Exception as e:
            print(f"Error initializing AI model: {str(e)}")
            self.model = None
            self.tokenizer = None

    def generate_response(self, question, learning_style):
        """Generate a response using the AI model based on learning style"""
        if not self.model or not self.tokenizer:
            return "I apologize, but the AI model is not properly initialized."

        try:
            # Create specialized prompts based on learning style
            if learning_style == "real_life":
                system_prompt = (
                    "You are a helpful tutor who explains concepts using real-life examples. "
                    "Make your explanations engaging and connected to daily experiences."
                )
            elif learning_style == "theoretical":
                system_prompt = (
                    "You are a helpful tutor who provides clear theoretical explanations. "
                    "Break down complex concepts while maintaining academic accuracy."
                )
            else:  # simplified
                system_prompt = (
                    "You are a helpful tutor who explains concepts in the simplest possible terms. "
                    "Use basic concepts and multiple simple examples."
                )

            # Format the prompt following Falcon's instruction format
            prompt = f"""System: {system_prompt}
Human: {question}
Assistant: Let me help you with that."""

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=1024,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode and clean up response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Assistant:")[-1].strip()
                
            return response

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error generating the response."

    def process_audio(self):
        """Listen for and process audio input from the microphone"""
        if not self.audio_available:
            return self.process_text()
            
        try:
            with sr.Microphone() as source:
                print("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Ready! Please ask your question...")
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
                
                try:
                    text = self.recognizer.recognize_google(audio)
                    return text
                except sr.UnknownValueError:
                    print("Sorry, I couldn't understand that. Could you please repeat?")
                    self.speak_response("I couldn't understand that. Could you please repeat?")
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
                    self.speak_response("Sorry, I'm having trouble processing your speech. Please try again.")
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
        return None

    def process_text(self):
        """Get input from text when audio is not available"""
        try:
            user_input = input("\nEnter your question (or 'exit' to quit): ")
            return user_input
        except Exception as e:
            print(f"Error getting text input: {str(e)}")
            return None

    def speak_response(self, text):
        """Convert text to speech and play it"""
        if self.audio_available:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"Error speaking response: {str(e)}")
        print(f"\nResponse: {text}")

class StudentProfile(PeopleDataManager):
    VALID_STYLES = {
        "real_life": "Explain with real-life examples",
        "theoretical": "Explain theoretically and in simple terms",
        "simplified": "Explain in the simplest terms with many examples"
    }
    
    def __init__(self, data_directory='people_database', excel_path=None):
        super().__init__(data_directory)
        self.learning_styles = {}
        self.excel_path = excel_path
        if self.excel_path:
            self.load_learning_styles_from_excel()
        else:
            print("No Excel file provided. Starting with an empty dictionary.")

    def _get_safe_name(self, name):
        """
        Convert name to a safe format for file system operations
        """
        return name.lower().replace(" ", "_")

    def load_learning_styles_from_excel(self):
        """
        Load learning styles from an Excel file if provided.
        """
        try:
            if not os.path.exists(self.excel_path):
                print(f"Excel file not found: {self.excel_path}")
                return

            df = pd.read_excel(self.excel_path)
            valid_styles = set(self.VALID_STYLES.keys())

            for _, row in df.iterrows():
                name = row['name']
                style = row['learning_style'].lower()

                if style not in valid_styles:
                    print(f"Warning: Invalid learning style '{style}' for {name}. "
                          f"Valid styles are: {', '.join(valid_styles)}")
                    continue

                safe_name = self._get_safe_name(name)
                self.learning_styles[safe_name] = {
                    'style': style,
                    'last_updated': datetime.now().isoformat()
                }

            print(f"Successfully loaded learning styles for {len(self.learning_styles)} students from Excel.")
        except Exception as e:
            print(f"Error loading learning styles from Excel: {str(e)}")

class EnhancedObjectDetector(NextGenObjectDetector):
    def __init__(self, image_directory="C:\\Users\\sharo\\OneDrive\\Desktop\\Research\\Object detection - Research\\image_directory", 
                 object_detection_model='yolov8x.pt'):
        super().__init__(object_detection_model)
        
        self.image_directory = image_directory
        self.student_profiles = None
        self.ai_tutor = AITutor()
        self.is_running = False
        self.current_student = None
        self.detection_confirmed = False
        self.confirmation_frames = 0
        self.REQUIRED_CONFIRMATION_FRAMES = 5  # Reduced from 10 to make detection faster
        self.name_mapping = {}
        
        self.known_face_encodings = []
        self.known_face_names = []
        
        self.load_faces_from_directory()

    def create_name_mapping(self):
        """Create mapping between folder names and Excel names"""
        if not self.student_profiles or not self.student_profiles.learning_styles:
            print("Warning: No student profiles loaded")
            return

        print("\nCreating name mappings...")
        self.name_mapping = {}
        excel_names = list(self.student_profiles.learning_styles.keys())
        
        for folder_name in set(self.known_face_names):
            std_folder_name = self.standardize_name(folder_name)
            
            for excel_name in excel_names:
                std_excel_name = self.standardize_name(excel_name)
                if std_folder_name == std_excel_name:
                    self.name_mapping[folder_name] = excel_name
                    print(f"Mapped '{folder_name}' to '{excel_name}'")
                    break
            
            if folder_name not in self.name_mapping:
                print(f"Warning: No match found for '{folder_name}'")
                print(f"Available names in Excel: {', '.join(excel_names)}")
                
    def standardize_name(self, name):
        """Standardize name format for comparison"""
        name = os.path.splitext(name)[0]
        name = name.replace('_', ' ').replace('  ', ' ')
        return name.strip().lower()
    
    def load_faces_from_directory(self):
        """Load faces from the image directory"""
        try:
            print(f"\nLoading faces from: {self.image_directory}")
            
            if not os.path.exists(self.image_directory):
                print(f"Error: Image directory not found: {self.image_directory}")
                return

            # Clear existing data
            self.known_face_encodings = []
            self.known_face_names = []
            
            loaded_faces = 0
            for person_folder in os.listdir(self.image_directory):
                person_path = os.path.join(self.image_directory, person_folder)
                if os.path.isdir(person_path):
                    for image_file in os.listdir(person_path):
                        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            try:
                                image_path = os.path.join(person_path, image_file)
                                face_image = face_recognition.load_image_file(image_path)
                                face_encodings = face_recognition.face_encodings(face_image)
                                
                                if face_encodings:
                                    self.known_face_encodings.append(face_encodings[0])
                                    self.known_face_names.append(person_folder)
                                    loaded_faces += 1
                            except Exception as e:
                                print(f"Error processing {image_file}: {str(e)}")

            print(f"Total faces loaded: {loaded_faces}")
            
        except Exception as e:
            print(f"Error during face loading: {str(e)}")

    def detect_and_process(self, frame):
        """Detect faces and process learning styles"""
        if len(self.known_face_encodings) == 0:
            return frame, False, None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        student_detected = False
        current_style = None
        
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(
                    self.known_face_encodings,
                    face_encoding,
                    tolerance=0.6
                )
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    learning_style = self.get_learning_style(name)
                    
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, f"Name: {name}", (left, top - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    if learning_style:
                        cv2.putText(frame, f"Style: {learning_style}", 
                                  (left, bottom + 25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        current_style = learning_style
                    
                    if self.current_student == name:
                        self.confirmation_frames += 1
                        print(f"Confirming frames: {self.confirmation_frames}/{self.REQUIRED_CONFIRMATION_FRAMES}")
                    else:
                        self.current_student = name
                        self.confirmation_frames = 1
                    
                    if self.confirmation_frames >= self.REQUIRED_CONFIRMATION_FRAMES:
                        student_detected = True
                        self.detection_confirmed = True  # Set the flag here
                        
        else:
            self.confirmation_frames = 0
            self.current_student = None
        
        return frame, student_detected, current_style


    def get_learning_style(self, person_name):
        """Get learning style from Excel data"""
        if not self.student_profiles:
            return None
        
        try:
            excel_name = self.name_mapping.get(person_name)
            if excel_name in self.student_profiles.learning_styles:
                return self.student_profiles.learning_styles[excel_name]['style']
        except Exception as e:
            print(f"Error getting learning style: {str(e)}")
        return None

    def start_tutoring_session(self, learning_style):
        """Start tutoring session with detected learning style"""
        if not self.current_student or not learning_style:
            return

        print(f"\n=== Starting Tutoring Session ===")
        welcome_message = f"Hi {self.current_student}! I'm ready to help you learn. Shoot away your questions!"
        print(welcome_message)
        self.ai_tutor.speak_response(welcome_message)
        
        print("\nListening for your questions... (say 'exit' to stop)")
        
        while True:
            try:
                print("\nListening...")
                question = self.ai_tutor.process_audio()
                
                if question:
                    print(f"\nYou asked: {question}")
                    if question.lower() in ['exit', 'quit', 'stop']:
                        goodbye_message = f"Goodbye {self.current_student}! Have a great day!"
                        print(goodbye_message)
                        self.ai_tutor.speak_response(goodbye_message)
                        break
                        
                    response = self.ai_tutor.generate_response(question, learning_style)
                    print(f"\nResponse: {response}")
                    self.ai_tutor.speak_response(response)
            except Exception as e:
                print(f"Error during tutoring session: {str(e)}")
                continue

    def real_time_detection(self):
        """Run real-time detection with learning style integration"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\nInitializing camera... Please look at the camera.")
        
        try:
            while True:  # Changed from while not self.detection_confirmed
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read from camera")
                    break
                
                processed_frame, student_detected, learning_style = self.detect_and_process(frame)
                cv2.imshow('Face Recognition', processed_frame)
                
                if self.detection_confirmed and learning_style:  # Changed condition
                    print(f"\nStudent verification complete!")
                    print(f"Name: {self.current_student}")
                    print(f"Learning Style: {learning_style}")
                    
                    # Close the camera window
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    # Start the tutoring session immediately
                    self.start_tutoring_session(learning_style)
                    break
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nDetection cancelled by user")
                    break
                    
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()

# Main execution
if __name__ == '__main__':
    excel_path = r"C:\Users\sharo\OneDrive\Desktop\Research\Object detection - Research\Data.xlsx"
    image_dir = r"C:\Users\sharo\OneDrive\Desktop\Research\Object detection - Research\image_directory"
    
    # Create and initialize detector
    detector = EnhancedObjectDetector(image_directory=image_dir)
    
    # Load student profiles from Excel
    detector.student_profiles = StudentProfile(excel_path=excel_path)
    
    # Create name mapping
    detector.create_name_mapping()
    
    # Print current mappings
    print("\nCurrent name mappings:")
    for folder_name, excel_name in detector.name_mapping.items():
        print(f"Folder: '{folder_name}' -> Excel: '{excel_name}'")
    
    # Print Excel contents for debugging
    print("\nNames in Excel file:")
    for name in detector.student_profiles.learning_styles.keys():
        print(f"- {name}")
    
    # Start detection
    print("\nStarting face detection... Press 'q' to quit.")
    detector.real_time_detection()


