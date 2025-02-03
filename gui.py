import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import requests

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")

        # Интерфейс
        self.label = tk.Label(root, text="Выберите изображение:")
        self.label.pack()

        self.select_button = tk.Button(root, text="Выбрать файл", command=self.load_image)
        self.select_button.pack()

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Отображение изображения
            image = Image.open(file_path)
            image.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            # Отправка изображения в API
            with open(file_path, 'rb') as f:
                response = requests.post(
                    "http://127.0.0.1:5000/predict",
                    files={'image': f}
                )
            result = response.json()
            self.result_label.config(text=f"Результат: класс {result['predicted_name']}")

# Запуск приложения
root = tk.Tk()
app = FaceRecognitionApp(root)
root.mainloop()