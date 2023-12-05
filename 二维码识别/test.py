import cv2
from pyzbar.pyzbar import decode

img = cv2.imread('./1.png')

code = decode(img)

print(code)
for barcode in code:
    print(type(barcode.data))
    mes = barcode.data.decode('utf-8')
    print(type(mes))