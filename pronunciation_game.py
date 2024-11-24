import cv2
import speech_recognition as sr
from tensorflow.keras.preprocessing import image
import numpy as np
import jellyfish
import os
from transformers import pipeline
import google.generativeai as genai
from dotenv import load_dotenv
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import torch

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

class PronunciationGame:
    def __init__(self, root):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.recognizer = sr.Recognizer()
        self.camera = cv2.VideoCapture(0)
        self.gemini_api_key = api_key
        self.score = 0
        self.total_objects = 3
        self.attempts = 0
        self.hint_text = "" 
        self.root = root
        self.root.title("SpeakUp Pronunciation Game")
        
        self.root.attributes('-fullscreen', True)
        
        self.label = tk.Label(root, text="Position the item in the frame, then press 'Capture' when ready.", font=("Arial", 20))
        self.label.pack(pady=20)
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.canvas = tk.Canvas(root, width=screen_width, height=screen_height - 300)
        self.canvas.pack()

        self.control_frame = tk.Frame(root)
        self.control_frame.pack(pady=20, anchor='n')

        self.result_label = tk.Label(self.control_frame, text="", font=("Arial", 18))
        self.result_label.grid(row=0, column=0, columnspan=4, pady=10)

        self.capture_button = tk.Button(self.control_frame, text="Capture", font=("Arial", 16), command=self.capture_image)
        self.capture_button.grid(row=1, column=0, padx=10)

        self.pronounce_button = tk.Button(self.control_frame, text="Pronounce", font=("Arial", 16), state=tk.DISABLED, command=self.get_pronunciation)
        self.pronounce_button.grid(row=1, column=1, padx=10)

        self.reset_button = tk.Button(self.control_frame, text="Reset", font=("Arial", 16), command=self.reset_game)
        self.reset_button.grid(row=1, column=2, padx=10)

        self.exit_button = tk.Button(self.control_frame, text="Exit Fullscreen", font=("Arial", 16), command=self.exit_fullscreen)
        self.exit_button.grid(row=1, column=3, padx=10)

        self.score_label = tk.Label(self.control_frame, text=f"Score: {self.score}/{self.total_objects}", font=("Arial", 18))
        self.score_label.grid(row=2, column=0, columnspan=4, pady=10)
        
        self.video_loop()



    def capture_image(self):
        """Capture the current frame and identify the object."""
        ret, frame = self.camera.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image")
            return

        self.object_name = self.identify_object(frame)
        if self.object_name:
            self.label.config(text=f"I see a {self.object_name}! Please pronounce it.")
            self.pronounce_button.config(state=tk.NORMAL)
            self.attempts = 0  

    def identify_object(self, frame):
        """Identify object in the image using YOLOv5."""
        results = self.model(frame)
        
        labels = results.pandas().xyxy[0]['name'].tolist()
        if labels:
            return labels[0]
        else:
            messagebox.showinfo("Info", "No object detected. Please try again.")
            return None
        
    def video_loop(self, scale_factor=0.8):
        """Continuously update the canvas with the camera feed while maintaining aspect ratio and scaling down."""
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            original_height, original_width = frame.shape[:2]

            canvas_width = max(self.canvas.winfo_width(), 1)
            canvas_height = max(self.canvas.winfo_height(), 1)

            aspect_ratio = original_width / original_height

            available_height = canvas_height - 100  # Deduct space for hint text

            if canvas_width / available_height > aspect_ratio:
                new_height = int(available_height * scale_factor)
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = int(canvas_width * scale_factor)
                new_height = int(new_width / aspect_ratio)

            if new_width > 0 and new_height > 0:
                frame_resized = cv2.resize(frame, (new_width, new_height))
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)

                self.canvas.delete("all")
                self.canvas.create_image((canvas_width - new_width) // 2, 
                                         (canvas_height - new_height - 100) // 2, 
                                         anchor="nw", image=imgtk)
                self.canvas.imgtk = imgtk  


        self.root.after(10, self.video_loop)

    def get_pronunciation(self):
        """Get user's pronunciation and compare it."""
        with sr.Microphone() as source:
            self.label.config(text="Please pronounce the word...")
            try:
                audio = self.recognizer.listen(source, timeout=5)
                spoken_word = self.recognizer.recognize_google(audio).lower()

                if self.compare_pronunciations(self.object_name, spoken_word):
                    self.result_label.config(text="🎉 Perfect pronunciation!", fg="green")
                    self.score += 1
                    self.update_score()
                    self.label.config(text="Press 'Capture' for the next object.")
                    self.pronounce_button.config(state=tk.DISABLED)
                else:
                    self.attempts += 1
                    self.result_label.config(text="❌ Not quite right. Try again!", fg="red")

                    if self.attempts == 2:
                        hints = self.find_similar_words(self.object_name)
                        hint_text = "\n".join(hints)
                        self.hint_text = f"Hint: {hint_text}" 
                        self.result_label.config(text=f"Hint: {hint_text}", fg="orange")

            except sr.UnknownValueError:
                self.result_label.config(text="Sorry, I couldn't understand. Try again.", fg="red")
            except sr.WaitTimeoutError:
                self.result_label.config(text="Timeout. Please try again.", fg="red")



    def compare_pronunciations(self, target_word, spoken_word):
        """Compare target word with spoken word using phonetic matching."""
        if not spoken_word:
            return False
        similarity = jellyfish.jaro_winkler_similarity(target_word.lower(), spoken_word.lower())
        return similarity > 0.8

    def find_similar_words(self, word):
        prompt = f"Generate 3 words phonetically similar to {word}."
        response = model.generate_content(prompt)
        hints = [part.text for part in response._result.candidates[0].content.parts]
        return hints

    def update_score(self):
        """Update the score display."""
        self.score_label.config(text=f"Score: {self.score}/{self.total_objects}")

        if self.score == self.total_objects:
            self.result_label.config(text="🎉 Congratulations, you won!", fg="blue")
            self.after_game_reset()

    def reset_game(self):
        """Reset the game to its initial state."""
        self.score = 0
        self.attempts = 0
        self.update_score()
        self.result_label.config(text="")
        self.label.config(text="Position the item in the frame, then press 'Capture' when ready.")
        self.capture_button.config(state=tk.NORMAL)
        self.pronounce_button.config(state=tk.DISABLED)

    def exit_fullscreen(self):
        """Exit fullscreen mode."""
        self.root.attributes('-fullscreen', False)

    def on_close(self):
        """Cleanup on closing the window."""
        self.camera.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    game = PronunciationGame(root)
    root.protocol("WM_DELETE_WINDOW", game.on_close)
    root.mainloop()
