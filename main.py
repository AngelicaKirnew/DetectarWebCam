#pip install opencv-python para instalar a biblioteca cv

import cv2

haarRosto = cv2.CascadeClassifier('haarcascade\\haarcascade_frontalface_default.xml')
# váriavel webcam recebe o cv2.VideoCapture() que vai pegar qual camera do seu equipamento vamos usar
webcam = cv2.VideoCapture(0)

while True:
    #webcam.read() devolve 2 informações se tá conectado e a outra pega a informação da camera
    conectado, imagem = webcam.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    rostoEncontrados = haarRosto.detectMultiScale(imagemCinza,
                                                  scaleFactor=1.1,
                                                  minNeighbors=3,
                                                  minSize=(50,50),
                                                  maxSize=(200,200)
                                                  )
    for (x, y, largura, altura) in rostoEncontrados:
        cv2.rectangle(imagem, (x,y), (x+largura, y+altura), (35,64,208), 4)
    #dentro desse ord coloca uma tecla para sair da webcam
    cv2.waitKey(1)
    cv2.imshow("webcam", imagem)
    if (cv2.waitKey(1) == ord('q')):
        break

webcam.release()
cv2.destroyAllWindows()