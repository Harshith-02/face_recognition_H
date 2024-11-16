import pickle

pickle_file_path = r'C:\Users\harsh\OneDrive\Desktop\face_recognition\face_encodings.pkl'

with open(pickle_file_path, 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

print("Pickle file loaded successfully!")
print(f"Known faces: {known_face_names}")
