from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
import time
import cv2



def detect_burger(request):
    if request.method == 'POST' and request.FILES['screenshot']:
        screenshot = request.FILES['screenshot']
        fs = FileSystemStorage()
        filename = fs.save(screenshot.name, screenshot)
        uploaded_file_url = fs.url(filename)
        full_path = os.path.join(settings.BASE_DIR, uploaded_file_url[1:])
        start_time = time.time()
        result_img = detect_burger_menu(full_path)
        end_time = time.time()
        elapsed_time = end_time - start_time
        return render(request, 'result.html', {
            'uploaded_file_url': uploaded_file_url,
            'result_img': result_img,
            'elapsed_time': elapsed_time
        })
    return render(request, 'index.html')


def detect_burger_menu(full_path):
    img = cv2.imread(full_path)
    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    top_thresh = int(height * 0.4)
    contours, hierarchy = cv2.findContours(thresh[:top_thresh, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    file_path, file_name = os.path.split(full_path)
    file_name, file_ext = os.path.splitext(file_name)
    save_path = os.path.join(file_path, 'red_menu', f'{file_name}_red{file_ext}')

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    cv2.imwrite(save_path, img)
    return img