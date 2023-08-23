import cv2
import os
import numpy as np
import time

# Load the Haarcascades face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a directory to store face images if not exists
if not os.path.exists('data'):
    os.makedirs('data')

def save_face(name, face):
    # Save the face image to the 'data' directory
    face_path = os.path.join('data', name, f'{name}_{len(os.listdir(os.path.join("data", name))) + 1}.jpg')
    os.makedirs(os.path.dirname(face_path), exist_ok=True)
    cv2.imwrite(face_path, face)
    print(f"Face saved as {face_path}")


def add_face():
    name = input("Enter the person's name: ")
    person_dir = os.path.join('data', name)

    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    capture = cv2.VideoCapture(0)  # Open the camera

    save_interval = 1  # Time interval between saving each photo (in seconds)
    last_save_time = time.time() - save_interval  # Initialize last saved time

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        # Detect faces in the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y + h, x:x + w]
                save_face(name, face_roi)
                last_save_time = current_time

        cv2.imshow("Add Face", frame)
        if cv2.waitKey(1) == 27:  # Press the Esc key to exit the loop
            break

    capture.release()
    cv2.destroyAllWindows()

def detect_and_recognize():
    employees = []
    label_map = {}  # Create a mapping from names to labels
    label_idx = 0

    for person_name in os.listdir('data'):
        person_dir = os.path.join('data', person_name)
        if os.path.isdir(person_dir):
            label_map[label_idx] = person_name
            person_images = []
            for filename in os.listdir(person_dir):
                image_path = os.path.join(person_dir, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                person_images.append(image)
                employees.append((label_idx, image))  # Append label and image tuple
            label_idx += 1

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    labels = [label for label, _ in employees]  # Extract labels from the employee list
    images = [image for _, image in employees]  # Extract images from the employee list

    face_recognizer.train(images, np.array(labels))

    capture = cv2.VideoCapture(0)  # Open the camera

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray_frame[y:y + h, x:x + w], (100, 100))

            label, confidence = face_recognizer.predict(face_roi)
            if confidence < 70:  # Adjust this threshold based on your dataset
                name = label_map[label]
                cv2.putText(frame, f"Hello {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)
            else:
                cv2.putText(frame, "Who are you?", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Detect & Recognize", frame)
        if cv2.waitKey(1) == 27:  # Press the Esc key to exit the loop
            break

    capture.release()
    cv2.destroyAllWindows()



def main():
    while True:
        print("Choose an option:")
        print("1. Add a Face")
        print("2. Detect & Recognize Face")
        choice = input("Enter 1 or 2: ")

        if choice == '1':
            add_face()
        elif choice == '2':
            detect_and_recognize()
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()