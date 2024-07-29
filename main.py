import json
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from pytesseract import pytesseract


def find_element(element, screen, windows_coords):
    """
    find_element ищет элемент на найденном окне.

    :param element: элемент для нахождения
    :param screen: самое изображение экрана
    :param windows_coords: координаты найденного окна
    :return: изображение с найденным элементом и координаты найденного элемента
    """
    img_rgb = screen
    assert img_rgb is not None, "file could not be read, check with os.path.exists()"
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = element
    assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    x, y, w_window, h_window = windows_coords
    h_window //= 10
    loc = np.where(res >= threshold)
    coord_fin = 0
    for pt in zip(*loc[::-1]):
        center_x = int(pt[0] + w / 2)
        center_y = int(pt[1] + h / 2)
        # Проверка: принадлежит ли найденный элемент окну
        if x <= center_x <= x + w_window and y <= center_y <= y + h_window:
            cv2.circle(img_rgb, (center_x, center_y), int(max(w, h) / 2), (0, 0, 255), 2)
            coord_fin = center_x

    return img_rgb, coord_fin


def find_buttons_in_window(screenshot, window_coords, btn_hide, btn_ful_screen, btn_exit, icon):
    """
    find_buttons_in_window ищет нужные кнопки на окне.

    :param screenshot: исходный скриншот экрана
    :param window_coords: координаты найденного окна
    :param btn_hide: кнопка сокрытия
    :param btn_ful_screen: кнопка полного экрана
    :param btn_exit: кнопка закрытия
    :param icon: иконка документа
    :return: итоговое изображение с отмеченными элементами и координаты всех элементов
    """
    # Выделите окно на скриншоте
    x, y, w, h = window_coords
    cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 255, 0), 2)
    elements = [icon, btn_hide, btn_ful_screen, btn_exit]
    screen_with_btn = screenshot
    coord_all = []
    for j in range(len(elements)):
        screen_with_btn, coord_el = find_element(elements[j], screenshot, window_coords)
        coord_all.append(coord_el)
    # Возвращаем координаты финального изображение
    return screen_with_btn, coord_all


def read_window_title(screenshot_path, window_coords):
    """
    read_window_title ищет заголовок документа.

    :param screenshot_path: исходный скриншот
    :param window_coords: координаты окна, в котором искать заголовок
    :return: найденный текст и его координаты
    """
    x, y, w, h = window_coords
    # Загружаем скриншот
    screenshot = cv2.imread(screenshot_path)
    # Вычисляем координаты области заголовка (предположим, что высота заголовка 30 пикселей)
    title_coords = (x+25, y, 200, 30)
    # Вырезаем область заголовка
    title_image = screenshot[title_coords[1]:title_coords[1] + title_coords[3],
                  title_coords[0]:title_coords[0] + title_coords[2]]
    scale_factor = 4  # Увеличиваем изображение в 4 раза
    title_image = cv2.resize(title_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    # Преобразуем в оттенки серого
    gray = cv2.cvtColor(title_image, cv2.COLOR_BGR2GRAY)
    # Повышаем контрастность
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)
    # Удаляем шум
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    opened = cv2.morphologyEx(contrast_enhanced, cv2.MORPH_OPEN, kernel, iterations=1)
    text = pytesseract.image_to_string(opened, config='--psm 6', lang='rus+eng')

    return text.strip(), title_coords[:2]


def find_template_matching(template, contours, grey):
    """
        find_template_matching функция для нахождения контура по находящейся внутри элемент.

        :param template: элемент внутри контура
        :param contours: все контуры
        :param grey: отфильтрованное исходное изображение
        :return: координаты нужного контура
        """
    # Маска для иконки
    mask = np.zeros(template.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (0, 0), (template.shape[1], template.shape[0]), 255, -1)

    # Использование метода поиска шаблона (template matching)
    best_match = False
    min_distance = 90000
    for c in contours:
        # Получение области контура
        x, y, w, h = cv2.boundingRect(c)
        # Проверка, чтобы шаблон был меньше или равен ROI
        if template.shape[0] <= h and template.shape[1] <= w:
            ROI = grey[y:y + h, x:x + w]

            result = cv2.matchTemplate(ROI, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # Установка порога соответствия
            threshold = 0.8
            if maxVal >= threshold and (best_match is False or maxVal > best_match[0]) and (maxVal <= 1.0):
                print(maxVal)
                center_x = x + w
                center_y = y + h
                distance = ((center_x - result.shape[1]) + (center_y - result.shape[0]))
                if distance < min_distance:
                    min_distance = distance
                    best_match = (maxVal, x, y, w, h)
                    print("Найден контур!")

    if best_match:
        _, x, y, w, h = best_match
        return x, y, w, h
    else:
        return None


def find_window(screenshot_path, icon_path, thresh):
    """
        find_window ищет нужное окно документа.

        :param screenshot_path: исходный скриншот экрана
        :param icon_path: исходная иконка текстового документа
        :param thresh: порог для исходного изображения
        :return: координаты нужного контура
        """
    screenshot = cv2.imread(screenshot_path)
    img_grey = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    # Порог и обработка изображения
    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    # Поиск контуров
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Загрузка шаблона иконки
    template = cv2.imread(icon_path, cv2.IMREAD_GRAYSCALE)

    return find_template_matching(template, contours, img_grey)


if __name__ == '__main__':
    icon_path = 'icon_new.png'
    full_screen = str(input("Документ открыт во весь экран? (введите y - если да и n - если в окне)"))
    thresh = 180
    if full_screen == 'y':
        btn_full_screen = cv2.imread('full_screen.png', cv2.IMREAD_GRAYSCALE)
    elif full_screen == 'n':
        btn_full_screen = cv2.imread('btn_full_screen.jpg', cv2.IMREAD_GRAYSCALE)
    else:
        print("Неверный ввод")
        quit()

    number_test = int(input("Выберите тест (1-5)"))
    if 1 <= number_test <= 5:
        if number_test == 3:
            thresh = 229
        screenshot_path = "test_" + str(number_test) + ".png"
        path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        pytesseract.tesseract_cmd = path_to_tesseract

        window_coords = find_window(screenshot_path, icon_path, thresh)

        if window_coords:
            icon = cv2.imread(icon_path, cv2.IMREAD_GRAYSCALE)
            btn_hide_2 = cv2.imread('btn_hide_full_screen.png', cv2.IMREAD_GRAYSCALE)
            btn_exit_2 = cv2.imread('btn_full_screen_exit.png', cv2.IMREAD_GRAYSCALE)
            x, y, w, h = window_coords
            print(f'Найдено окно с координатами: x={x}, y={y}, w={w}, h={h}')

            screenshot = cv2.imread(screenshot_path)
            cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 255, 0), 2)
            screen_fin, coord_all_el = find_buttons_in_window(screenshot, window_coords, btn_hide_2, btn_full_screen, btn_exit_2, icon)
            # Извлекаем текст заголовка
            window_title, coords_title = read_window_title(screenshot_path, window_coords)

            if window_title:
                print(f"Текст заголовка окна: {window_title}")
                org = (coords_title[0] + 30, coords_title[1])
                pil_im = Image.fromarray(screen_fin)
                draw = ImageDraw.Draw(pil_im)
                font = ImageFont.truetype("arial.ttf", 30)
                # Добавление текста
                draw.text(org, str(window_title), font=font, fill=(255, 0, 0))
                cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            else:
                print("Текст заголовка не найден.")
            #  Создаем словарь для хранения данных
            data = {
                "window_tl": x,
                "window_br": x + w,
                "caption": str(window_title),
                "btn_minimize": coord_all_el[1],
                "btn_maximize": coord_all_el[2],
                "btn_close": coord_all_el[3],
                "cv2_im_processed": 'C:\\Project\\Test_Sin\\result.png'
            }
            #  Преобразуем словарь в JSON-строку
            json_string = json.dumps(data, ensure_ascii=False)

            #  Сохраняем JSON-строку в файл
            with open("data.json", "w", encoding='utf-8') as f:
                f.write(json_string)

            cv2.imshow('Result', cv2_im_processed)
            cv2.waitKey(0)
        else:
            print('Окно не найдено.')
    else:
        print("Неверный номер теста")
