from tkinter import *
from tkinter import filedialog
from PIL import Image
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
import threading

# Initialize the main tkinter window
main = Tk()
main.title("Object Tracking Using DeepLearning")
main.state('zoomed')

# Load YOLOv8 model
model = YOLO('yolo11s.pt')

# Text-to-speech for detected objects
# Initialize the engine globally
engine = pyttsx3.init()

def stop_voice():
    """Stop the voice immediately."""
    engine.stop()
    print("Voice stopped.")

def voice(label):
    """Function to speak the label."""
    def speak():
        engine.say(label)
        engine.runAndWait()

    # Start the speaking in a separate thread
    voice_thread = threading.Thread(target=speak)
    voice_thread.start()

# Helper function to count and format detected objects
def count_objects(results):
    detected_objects = {}
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            detected_objects[label] = detected_objects.get(label, 0) + 1
    return detected_objects

def format_results1(detected_objects):
    result_text = "Detected Objects:\n"
    for obj, count in detected_objects.items():
        result_text += f"{obj}: {count}\n"
    return result_text

# Function to draw bounding boxes and labels on the frame
def draw_boxes(frame, results):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            confidence = box.conf[0]

            # Draw rectangle for the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label and confidence
            label_text = f"{label} {confidence:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Function to handle image upload
def uploadImage():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="images", title="Select an Image")
    if not filename:
        text.insert(END, "No file selected.\n")
        return

    img = cv2.imread(filename)

    # Perform inference
    results = model(img)

    # Count objects and format results
    detected_objects = count_objects(results)
    result_text = format_results1(detected_objects)

    # Display results
    text.insert(END, result_text)
    
    # Start voice in a separate thread
    voice(result_text)
    # Draw bounding boxes and labels
    annotated_img = draw_boxes(img, results)
    cv2.imshow("Image Detection", annotated_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to format the results
def format_results(results):
    detected_objects = {}
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            detected_objects[label] = detected_objects.get(label, 0) + 1
    result_text = "\n".join([f"{obj}: {count}" for obj, count in detected_objects.items()])
    return result_text


def uploadVideo():
    text.delete('1.0', END)  # Clear the text area
    video_path = filedialog.askopenfilename(initialdir="videos")  # Open file dialog to select video
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Perform inference
        results = model(frame_pil)

        # Format results as text
        result_text = format_results(results)

        # Display results in the text widget
        text.insert('end', result_text + "\n")

        # Draw bounding boxes on the frame
        annotated_frame = draw_boxes(frame, results)

        # Display the annotated frame
        cv2.imshow('Annotated Frame', annotated_frame)

        # Start voice in a separate thread
        voice(result_text)

        # Check if 'q' is pressed to exit the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_voice()  # Stop the voice when 'q' is pressed
            break

    cap.release()
    cv2.destroyAllWindows() 
            

# Webcam video capture function
def webcamVideo():
    text.delete('1.0', END)  # Clear the text area
    cap = cv2.VideoCapture(0)  # Open webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Perform inference
        results = model(frame_pil)

        # Format results as text
        result_text = format_results(results)

        # Display results in the text widget
        text.insert('end', result_text + "\n")

        # Draw bounding boxes on the frame
        annotated_frame = draw_boxes(frame, results)

        # Display the annotated frame
        cv2.imshow('Annotated Frame', annotated_frame)

        # Start voice in a separate thread
        voice(result_text)

        # Check if 'q' is pressed to exit the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_voice()  # Stop the voice when 'q' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()            

                

def exitApp():
    main.destroy()

# GUI Layout
title = Label(main, text="Object Detection Using Deep Learning", font=('Arial', 20, 'bold'), bg='lightblue1', fg='black')
title.pack(pady=20)

button_frame = Frame(main,bg='lightblue1')
button_frame.pack(pady=20)

upload_image_btn = Button(button_frame, text="Upload Image", font=('Helvetica', 14), bg='magenta', fg='white', command=uploadImage)
upload_image_btn.grid(row=0, column=0, padx=20)

upload_video_btn = Button(button_frame, text="Upload Video", font=('Helvetica', 14), bg='maroon1', fg='white', command=uploadVideo)
upload_video_btn.grid(row=0, column=1, padx=20)

webcam_btn = Button(button_frame, text="Start Webcam", font=('Helvetica', 14), bg='mediumseagreen', fg='white', command=webcamVideo)
webcam_btn.grid(row=0, column=2, padx=20)

exit_btn = Button(button_frame, text="Exit", font=('Helvetica', 14), bg='red', fg='white', command=exitApp)
exit_btn.grid(row=0, column=3, padx=20)

text = Text(main, height=30, width=90, font=('Georgia', 14))
text.pack(pady=20)

main.config(bg='lightblue1')
main.mainloop()
