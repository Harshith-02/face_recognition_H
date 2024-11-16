import face_recognition
import os
import pickle

known_face_encodings = []
known_face_names = []

# Path to the directory containing known face images
known_faces_dir = r"C:\Users\harsh\OneDrive\Desktop\face_recognition\known_faces"

for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Ensure only images are processed
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            encoding = encodings[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])

# Save the encodings and names to a file in the same directory
pickle_file_path = r'C:\Users\harsh\OneDrive\Desktop\face_recognition\face_encodings.pkl'
with open(pickle_file_path, 'wb') as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print(f"Encodings saved to {pickle_file_path}")
