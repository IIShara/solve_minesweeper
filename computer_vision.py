import cv2 as cv
import numpy as np
import os
import tkinter as tk
from PIL import ImageGrab

TEMPLATES_DIR = "templates"

def overlay_game_area(area):
    """
    Отображает оверлей вокруг выбранной области
    """
    x1, y1, x2, y2 = area
    width = x2 - x1
    height = y2 - y1

    # Создаем прозрачное окно
    root = tk.Tk()
    root.overrideredirect(True)  # Убираем рамку окна
    root.geometry(f"{width}x{height}+{x1}+{y1}")
    root.lift()
    root.attributes("-topmost", True)
    root.attributes("-transparentcolor", "black")  # Черный цвет станет прозрачным

    # Canvas для рисования рамки
    canvas = tk.Canvas(root, width=width, height=height, highlightthickness=0, bd=0, bg="black")
    canvas.pack()

    # Рисуем рамку
    canvas.create_rectangle(0, 0, width, height, outline='red', width=3)

    # Обновляем окно в потоке
    def update_overlay():
        if not stop_solver:
            root.after(100, update_overlay)
        else:
            root.destroy()

    update_overlay()
    root.mainloop()

def find_minesweeper_field(template_path=os.path.join(TEMPLATES_DIR, "field_template.png"), threshold=0.9):
    """
    Ищет область игрового поля Сапёра на экране
    :param template_path: путь к шаблону поля
    :param threshold: порог совпадения
    :return: (x1, y1, x2, y2) или None
    """
    if not os.path.exists(template_path):
        print(f"[ERROR] Шаблон не найден: {template_path}")
        return None

    # Загружаем шаблон
    template = cv.imread(template_path, 0)
    tw, th = template.shape[::-1]

    # Делаем скриншот всего экрана
    screen_img = np.array(ImageGrab.grab())
    screen_gray = cv.cvtColor(screen_img, cv.COLOR_RGB2GRAY)

    # Ищем совпадение
    res = cv.matchTemplate(screen_gray, template, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    matches = list(zip(*loc[::-1]))
    if matches:
        top_left = matches[0]
        bottom_right = (top_left[0] + tw, top_left[1] + th)
        print(f"[SUCCESS] Поле найдено автоматически: {top_left} → {bottom_right}")

        # === ВИЗУАЛИЗАЦИЯ НА ЭКРАНЕ ===
        field_coords = (*top_left, *bottom_right)
        overlay_game_area(field_coords)

        return field_coords

    print("[INFO] Поле Сапёра не найдено")
    return None