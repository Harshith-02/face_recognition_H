import face_recognition
import cv2
import os

known_face_encodings = []
known_face_names = []

# Path to the directory containing known face images
known_faces_dir = r"C:\Users\harsh\OneDrive\Desktop\face_recognition\known_faces"

# Load and encode known faces
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Ensure only images are processed
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            encoding = encodings[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])

# Path to the image for face recognition
image_path = r"C:\Users\harsh\OneDrive\Desktop\face_recognition\known_faces\person1.jpg"

# Load the image
img = cv2.imread(image_path)
if img is None:
    print("Failed to load image. Please check the file path and try again.")
else:
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the image
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    # Loop through each face found in the image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, use the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (left + 6, bottom - 6), font, 0.75, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow("Recognized Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
