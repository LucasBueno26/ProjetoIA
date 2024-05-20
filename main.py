import cv2
import numpy as np
import os
import threading
import tkinter as tk
from keras.models import model_from_json
from tkinter import ttk
from reportlab.lib.pagesizes import letter
from datetime import datetime
from reportlab.pdfgen import canvas
from io import BytesIO
from PIL import Image
from reportlab.lib.utils import ImageReader

'''
Fer2013 Dataset
'''

#Emoções definidas e treinadas pelo modelo
emotion_dict = {0: "Bravo", 1: "Nojo", 2: "Medo", 3: "Feliz", 4: "Neutro", 5: "Triste", 6: "Surpreso"}

#Carregando o modelo e compilando para uso
json_file = open('CNN/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("CNN/emotion_model.h5")

print("Modelo carregado e compilado.")

#Iniciando a webcam
cap = cv2.VideoCapture(0)

# Variáveis para armazenar a emoção detectada e a imagem atual
current_emotion = "Nenhuma"
current_frame = None
click_records = [] 

def detectar_emocao():
    global current_emotion, current_frame

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break

        #Arquivo treinado de detecção de face
        face_detector = cv2.CascadeClassifier('CNN/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces                       :
            cv2.rectangle(frame, (x, y-30), (x+w, y+h+20), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            current_emotion = emotion_dict[maxindex]  
            cv2.putText(frame, current_emotion, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,205), 2, cv2.LINE_AA)

        current_frame = frame.copy()

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def clique_botao(button_number):
    global current_emotion, current_frame
    print(f"Botão {button_number} foi clicado! Emoção atual: {current_emotion}")
    if current_frame is not None:
        # Armazena a imagem como um objeto PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
        click_records.append((f"Botão {button_number} foi clicado. Emoção detectada: {current_emotion}", pil_image))

def gui():
    root = tk.Tk()
    root.title("Novo Produto Marketing") #Simulando um novo produto para ser testado
    root.geometry("400x400")

    style = ttk.Style()
    style.configure("TButton", font=("Arial", 12), padding=10)
    
    main_frame = ttk.Frame(root, padding="10 10 10 10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    label = ttk.Label(main_frame, text="Clique nos botões", font=("Arial", 16))
    label.grid(row=0, column=0, columnspan=2, pady=10)
    
    buttons = [
        ("Botão 1", 1),
        ("Botão 2", 2),
        ("Botão 3", 3),
        ("Botão 4", 4),
        ("Botão 5", 5),
        ("Botão 6", 6),
    ]
    
    for idx, (text, value) in enumerate(buttons):
        button = ttk.Button(main_frame, text=text, command=lambda num=value: clique_botao(num))
        button.grid(row=(idx // 2) + 1, column=idx % 2, padx=10, pady=10, sticky=(tk.W, tk.E))

    for child in main_frame.winfo_children():
        child.grid_configure(padx=10, pady=10)
    
    root.mainloop()

def salvar_pdf(filename="resultados.pdf"):
    if os.path.exists(filename):
        os.remove(filename)

    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    c.drawString(100, height - 50, "Relatórios das interações e a emoções")
    y = height - 80

    for record, pil_image in click_records:
        # Converter a imagem PIL para um objeto compatível com ReportLab
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img = ImageReader(img_buffer)

        # Verificar se há espaço suficiente para a imagem na página atual
        if y - 200 < 100:  # Verifica se há espaço para a imagem na página seguinte
            c.showPage()
            c.drawString(100, height - 50, "Relatórios das interações e a emoções")
            y = height - 80

        c.drawImage(img, 100, y - 180, width=250, height=150)
        c.drawString(100, y, record)
        c.drawString(100, y - 15, f"Horário {datetime.now().strftime('%H:%M:%S')}, Imagem:")
        y -= 200

    c.save()
    os.system(f'start {filename}')


emotions_thread = threading.Thread(target=detectar_emocao)
emotions_thread.start()

# Chamando os metódos
gui()
cap.release()
cv2.destroyAllWindows()
salvar_pdf()
