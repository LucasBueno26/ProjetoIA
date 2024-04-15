import mediapipe as mp
import cv2
import numpy as np

# Configurações de desenho
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
left_eye_points = list(range(23, 30))
right_eye_points = list(range(252, 260))

# Crie uma instância do objeto FaceMesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Inicialize a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame.")
        break

    # Processamento de imagem
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # Verifique se alguma face foi encontrada
    if results.multi_face_landmarks:
        # Marcar landmarks na imagem
        for face_landmarks in results.multi_face_landmarks:
            # Converter landmarks para um formato numpy
            landmarks = np.array([[lm.x, lm.y] for lm in face_landmarks.landmark])

            # Extrair pontos dos lábios
            upper_lip = np.mean(landmarks[61:65], axis=0)
            lower_lip = np.mean(landmarks[65:68], axis=0)
            
            #Extrair olho esquerdo
            lower_lip2 = np.mean(landmarks[23:27], axis=0)
            upper_lip2 = np.mean(landmarks[28:29], axis=0)

            #Extrair olho direito
            lower_lip3 = np.mean(landmarks[257:259], axis=0)
            upper_lip3 = np.mean(landmarks[252:256], axis=0)

            # Calcular a distância entre os lábios
            lip_distance = lower_lip[1] - upper_lip[1]
            
            #direito
            lip_distance2 = lower_lip2[1] - upper_lip2[1]
            #esq
            lip_distance3 = lower_lip3[1] - upper_lip3[1]

            # Se a distância normalizada for maior que o limiar, consideramos que a pessoa está sorrindo
            cv2.putText(frame, f"direito {lip_distance2}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"esquerdo {lip_distance3}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if lip_distance > -0.15 and lip_distance2 < 0.038 and lip_distance3 > -0.038:
                cv2.putText(frame, "Sorrindo", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Neutro", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Desenhar um quadrado fixo para encaixar o rosto
            # Coordenadas do quadrado
            x1, y1 = int(frame.shape[1] * 0.42), int(frame.shape[0] * 0.25)
            x2, y2 = int(frame.shape[1] * 0.68), int(frame.shape[0] * 0.75)
            # Desenhar o quadrado
           # Desenhar uma forma oval
            cv2.putText(frame, "Encaixe o rosto no circulo", (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.ellipse(frame, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, (155,51, 153), 2)

            # Desenhar landmarks na imagem
            mp_drawing.draw_landmarks(image=frame,
                                       landmark_list=face_landmarks,
                                       connections=mp_face_mesh.FACEMESH_TESSELATION,
                                       landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0),
                                                                                     thickness=1,
                                                                                     circle_radius=1),
                                       connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                                                        thickness=1,
                                                                                        circle_radius=1))
      
            # Iterar sobre os pontos e desenhar os números apenas para os pontos dos olhos
            for idx, landmark in enumerate(face_landmarks.landmark):
                height, width, _ = frame.shape
                x, y = int(landmark.x * width), int(landmark.y * height)

                # Desenhar o número apenas se for um ponto do olho
                if idx in left_eye_points or idx in right_eye_points:
                    cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


    # Mostra o frame resultante
    # Mostra o frame resultante
    cv2.imshow('Face Mesh', frame)

    # Sair do loop quando pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
