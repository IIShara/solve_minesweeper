import tkinter as tk
from datetime import datetime
from turtledemo.nim import COLOR

from PIL import ImageGrab, Image, ImageDraw, ImageTk
import numpy as np
import cv2
import time
import pyautogui
import random
import keyboard
import threading
from itertools import combinations
import pytesseract
from computer_vision import find_minesweeper_field
import os
from datetime import datetime
import queue
from collections import defaultdict
from functools import partial

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TEMPLATES_DIR = "templates"
# === Глобальные переменные ===
stop_solver = False
pause_solver = False
templates = {}  # templates['X'] = [template1, template2, ...]
LOG_LEVEL = 'INFO'
start_time = None  # ← время начала решения

COLOR_MAP = {
    '0': '\033[90m',   # Серый
    '1': '\033[94m',   # Ярко-синий
    '2': '\033[92m',   # Зелёный
    '3': '\033[93m',   # Жёлтый
    '4': '\033[33m',   # Оранжевый (нельзя через стандартные ANSI, но можно эмулировать)
    '5': '\033[35m',   # Фиолетовый
    '6': '\033[36m',   # Бирюзовый
    '7': '\033[91m',   # Красный
    '8': '\033[34m',   # Тёмно-синий
    'F': '\033[91m',   # Красный
    'X': '\033[30m',   # Чёрный
    '?': '\033[93m',  # Жёлтый (неопределённая ячейка)
    'C': '\033[93m',  # Жёлтый
    'M': '\033[94m',   # Ярко-синий
}
RESET = '\033[0m'
SUCCESS = '\033[92m'
ERROR = '\033[91m'
WARNING = '\033[93m'

board_changes = queue.Queue()
current_board = None
board_lock = threading.Lock()

def load_templates():
    """
    Загружает все шаблоны из папки templates/
    Имя файла должно быть в формате:
        X_1.png, X_2.png
        F_1.png, F_2.png
        1_1.png, 1_2.png
    """
    global templates
    templates.clear()

    if not os.path.exists(TEMPLATES_DIR):
        print(f"[ERROR] Папка шаблонов не найдена: {TEMPLATES_DIR}")
        return

    # Перебираем все файлы в папке
    for filename in os.listdir(TEMPLATES_DIR):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            full_path = os.path.join(TEMPLATES_DIR, filename)

            # Парсим имя файла: например, X_1.png → label = 'X', номер = 1
            if '_' in filename and '.' in filename:
                base_label = filename.split('_')[0]
                try:
                    # Загружаем как grayscale
                    img = cv2.imread(full_path, 0)
                    if img is None:
                        print(f"[ERROR] Не удалось прочитать изображение: {full_path}")
                        continue

                    # Добавляем в словарь по типу ('X', 'F', '1', ...)
                    if base_label not in templates:
                        templates[base_label] = []
                    templates[base_label].append(img)
                    # print(f"[INFO] Шаблон загружен: {base_label}")
                except Exception as e:
                    print(f"[ERROR] Неверное имя файла: {filename} | {e}")
            else:
                print(f"[WARNING] Имя файла не соответствует формату: {filename}")

def print_colored_board(board):
    # === Цвета для консоли ===
    COLOR_MAP = {
        '0': '\033[90m',   # Серый
        '1': '\033[94m',   # Ярко-синий
        '2': '\033[92m',   # Зелёный
        '3': '\033[93m',   # Жёлтый
        '4': '\033[33m',   # Оранжевый (нельзя через стандартные ANSI, но можно эмулировать)
        '5': '\033[35m',   # Фиолетовый
        '6': '\033[36m',   # Бирюзовый
        '7': '\033[91m',   # Красный
        '8': '\033[34m',   # Тёмно-синий
        'F': '\033[91m',   # Красный
        'X': '\033[30m',   # Чёрный
        '?': '\033[93m',  # Жёлтый (неопределённая ячейка)
    }
    RESET = '\033[0m'

    print("\n[DEBUG] Текущее состояние доски:")
    for row in board:
        line = ""
        for cell in row:
            color_code = COLOR_MAP.get(cell, RESET)  # Если не найден — белый
            line += f"{color_code}{cell}{RESET} "
        print("  ", line)
    print(RESET)
# ==============================
# === Выбор области экрана ====
# ==============================
class AreaSelectorApp:
    def __init__(self, callback, is_cell=False):
        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-alpha", 0.3)
        self.root.configure(bg='black')
        self.root.bind("<Button-1>", self.on_click)
        self.root.bind("<B1-Motion>", self.on_move)
        self.root.bind("<ButtonRelease-1>", self.on_release)

        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.start_x = self.start_y = 0
        self.rect = None
        self.selected_area = None
        self.callback = callback
        self.is_cell = is_cell

    def on_click(self, event):
        self.start_x, self.start_y = event.x_root, event.y_root

    def on_move(self, event):
        if self.rect:
            self.canvas.delete(self.rect)
        cur_x, cur_y = event.x_root, event.y_root
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, cur_x, cur_y, outline='red', width=3)

    def on_release(self, event):
        end_x, end_y = event.x_root, event.y_root
        self.selected_area = (self.start_x, self.start_y, end_x, end_y)
        self.root.quit()
        self.root.destroy()
        if self.callback:
            self.callback(self.selected_area, self.is_cell)

    def run(self):
        self.root.mainloop()

# ==============================
# === Распознавание ячеек ====
# ==============================
def cell_color_to_type(cell_img):
    global pause_solver

    img = cell_img.convert('RGB')
    img_array = np.array(img)

    h, w, _ = img_array.shape

    # === ОБРЕЗАЕМ КРАЯ (оставляем 80% в центре)
    border_percent = 0.3  # 50%
    border_h = int(h * border_percent)
    border_w = int(w * border_percent)

    center = img_array[border_h:h - border_h, border_w:w - border_w]  # Центральная область

    avg_color = np.mean(center, axis=(0, 1)).astype(int)  # RGB среднее

    # CELL_COLORS = {
    #     '0':        (172, 171, 172),
    #     'X':        (186, 187, 189),
    #     'F':        (175, 164, 166),
    #     '1':        (143, 142, 187),
    #     '2':        (123,159, 123),
    #     '3':        (195, 127, 128),
    #     '4':        (130, 128, 162),
    #     '5':        (161, 124, 126),
    #     '6':        (0, 0, 0),
    #     '7':        (0, 0, 0),
    #     '8':        (133, 154, 154),
    # }

    CELL_COLORS = {
        '0':        [(82, 63, 44),
                     (82, 62, 43),
                     (77, 58, 41)],
        'X':        [(82, 97, 30),
                     (95, 100, 35),
                     (107, 105, 41),
                     (117, 122, 46),
                     (116, 109, 44),


                     (80, 128, 157),
                     (38, 111, 154),
                     (129, 113, 31)],
        'F':        [(220, 153, 151),
                     (175,121,114),
                     (160, 130, 89),
                     (202, 143, 135),
                     (107, 123, 41)],
        'C':        [(180, 146, 62)],
        'M':        [(50, 129, 166)],
        # '1':        [(1, 1, 1)], #(94, 98, 44)
        # '2':        [(113, 115, 36)],
        '3':        [(116, 97, 26)],
        '4':        [(118, 81, 25)],
        '5':        [(122, 61, 23)],
        '6':        [(128, 53, 21)],
        '7':        [(133, 37, 26)],
    }


    def color_distance(c1, c2):
        return sum((a - b) ** 2 for a, b in zip(c1, c2))

    min_dist = float('inf')
    closest_type = '?'

    for label, color_list in CELL_COLORS.items():
        for color in color_list:
            dist = color_distance(avg_color, color)
            if dist < min_dist:
                min_dist = dist
                closest_type = label

    if min_dist > 100:  # Можно настроить порог
        # print(f"[DEBUG] Неопределённая ячейка | Цвет: {avg_color}, дистанция: {min_dist}")
        # return cell_ocr_to_type(cell_img)
        # show_unknown_color_dialog(cell_img)
        return '?'
    color_code = COLOR_MAP.get(closest_type, RESET)
    # print(f"[DEBUG][COLOR] Ячейка распознана как: {color_code}{closest_type}{RESET} (дистанция: {min_dist}) | Цвет: {avg_color}")
    return closest_type

def cell_template_to_type(cell_img):
    global pause_solver, stop_solver
    cell_array = np.array(cell_img.convert('RGB'))
    gray_cell = cv2.cvtColor(cell_array, cv2.COLOR_RGB2GRAY)

    h, w = gray_cell.shape
    pad_w, pad_h = int(w * 0.1), int(h * 0.1)  # 5% от края
    center = gray_cell[pad_h:h - pad_h, pad_w:w - pad_w]  # Центральная область (90%)

    best_label = '?'
    best_score = 0.0

    for label, template_list in templates.items():
        for template in template_list:
            # # Уменьшаем размер шаблона на 10%
            # scaled_template = cv2.resize(template, (int(template.shape[1] * 0.9), int(template.shape[0] * 0.9)))
            if gray_cell.shape[:2] != template.shape:
                continue

            result = cv2.matchTemplate(gray_cell, template, cv2.TM_CCOEFF_NORMED)
            score = result[0][0]

            if score > best_score and score > 0.8:
                best_score = score
                best_label = label

    CONFIDENCE_THRESHOLD = 0.8
    if best_score < CONFIDENCE_THRESHOLD:
        # print(f"[DEBUG] Неопределённая ячейка | Дистанция: {best_score:.2f}")
        return cell_color_to_type(cell_img)
        show_unknown_color_dialog(cell_img)
    color_code = COLOR_MAP.get(best_label, RESET)
    # print(f"[DEBUG][TEMPLATE] Ячейка распознана как: {color_code}{best_label}{RESET} (score={best_score:.2f})")
    return best_label

def cell_ocr_to_type(cell_img):
    """
    Распознаёт тип ячейки через OCR после предобработки
    :param cell_img: PIL.Image — изображение ячейки
    :return: тип ячейки ('1', '2', ..., '?')
    """
    # Конвертируем в NumPy массив
    img = np.array(cell_img.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Увеличиваем резкость
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

    # Бинаризация + инверсия
    _, binary = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Дополнительный шумоподавляющий фильтр
    binary = cv2.medianBlur(binary, 3)

    # === Отладка: покажем, что видит OCR ===
    # cv2.imshow("OCR Input", binary)
    # cv2.waitKey(0)

    # Распознаём цифры
    config = '--psm 10 --oem 3 -c tessedit_char_whitelist=012345678'
    text = pytesseract.image_to_string(binary, config=config).strip()

    if text.isdigit() and 1 <= int(text) <= 8:
        color_code = COLOR_MAP.get(text, RESET)
        print(f"[DEBUG][OCR] Ячейка распознана как: {color_code}{text}{RESET}")
        return text
    else:
        print(f"[DEBUG] OCR не смог распознать число: '{text}'")
        # show_unknown_color_dialog(cell_img)
        return '?'
# ==============================
# = Распознавание игрового поля =
# ==============================

def overlay_game_area(area, cell_size):
    """
    Отображает оверлей вокруг выбранной области
    """
    x1, y1, x2, y2 = area
    width = x2 - x1
    height = y2 - y1

    # Создаем прозрачное окно
    overlay_root = tk.Tk()
    overlay_root.overrideredirect(True)  # Убираем рамку окна
    overlay_root.geometry(f"{width}x{height}+{x1}+{y1}")
    overlay_root.lift()
    overlay_root.attributes("-topmost", True)
    overlay_root.attributes("-transparentcolor", "black")  # Черный цвет станет прозрачным

    # Canvas для рисования рамки
    canvas = tk.Canvas(overlay_root, width=width, height=height, highlightthickness=0, bd=0, bg="black")
    canvas.pack()

    # Рисуем рамку
    canvas.create_rectangle(0, 0, width, height, outline='red', width=3)

    # Рисуем сетку
    for row in range(0, height, cell_size):
        canvas.create_line(0, row, width, row, fill="red", width=1)

    for col in range(0, width, cell_size):
        canvas.create_line(col, 0, col, height, fill="red", width=1)

    # === Диалог подтверждения ===
    def on_confirm():
        print("[ACTION] Область подтверждена")
        overlay_root.destroy()
        callback(area)

    # Показываем диалог
    dialog = tk.Toplevel(overlay_root)
    dialog.title("Подтверждение области")
    dialog.geometry("300x150")
    dialog.transient(overlay_root)
    dialog.grab_set()

    tk.Label(dialog, text="Подтвердите игровое поле:", font=("Arial", 12)).pack(pady=10)
    tk.Label(dialog, text=f"({x1}, {y1}) → ({x2}, {y2})", font=("Courier", 10)).pack(pady=5)

    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=20)

    tk.Button(btn_frame, text="Продолжить", width=10, command=on_confirm).pack(side=tk.LEFT, padx=10)

    overlay_root.wait_window(dialog)  # Ждём ответа пользователя

def validate_area(area, board_height_count, board_width_count, cell_size):
    x1, y1, x2, y2 = area
    width = x2 - x1
    height = y2 - y1
    if abs(width - board_width_count * cell_size) > cell_size:
        return False
    if abs(height - board_height_count * cell_size) > cell_size:
        return False
    return True

def detect_cell_size(template_path='templates/X_dota.png'):
    """
    Определяет размер ячейки по шаблону
    :param template_path: путь к файлу с шаблоном ячейки
    :return: размер ячейки (int) или None
    """
    if not os.path.exists(template_path):
        print(f"[ERROR] Шаблон не найден: {template_path}")
        return None

    # Загружаем шаблон
    template = cv2.imread(template_path)
    if template is None:
        print(f"[ERROR] Не удалось загрузить шаблон: {template_path}")
        return None

    h, w = template.shape[:2]
    cell_size = min(h, w)

    print(f"[INFO] Размер ячейки определён: {cell_size}x{cell_size}")
    return cell_size

def find_and_click_image(template_file, threshold=0.8):
    template_path = os.path.join(TEMPLATES_DIR, template_file)
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template is None:
        print(f"[ERROR] Не удалось загрузить изображение: {template_path}")
        return False

    if template.shape[2] == 4:
        template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)

    screenshot = pyautogui.screenshot()
    screen = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        h, w = template.shape[:2]
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2
        return center_x, center_y
    else:
        print("Элемент не найден.")
        return False


def find_minesweeper_field(board_width_count, board_height_count, CELL_SIZE, template_dir=os.path.join(TEMPLATES_DIR, "fields"), threshold=0.9):
    """
   Перебирает несколько шаблонов игрового поля и возвращает первую найденную область
   :param template_dir: папка с шаблонами полей
   :param threshold: порог совпадения
   :return: координаты области (x1, y1, x2, y2), или None
   """
    if not os.path.exists(template_dir):
        print(f"[ERROR] Папка с шаблонами поля не найдена: {template_dir}")
        return None

    # Получаем список всех шаблонов в папке
    templates = [f for f in os.listdir(template_dir) if f.endswith(".png") or f.endswith(".jpg")]
    if not templates:
        print(f"[INFO] Нет шаблонов в папке {template_dir}")
        return None

    # Делаем скриншот всего экрана
    screen_img = np.array(ImageGrab.grab())
    screen_gray = cv2.cvtColor(screen_img, cv2.COLOR_RGB2GRAY)

    # Перебираем шаблоны
    for template_file in templates:
        template_path = os.path.join(template_dir, template_file)
        template = cv2.imread(template_path, 0)
        if template is None:
            print(f"[ERROR] Не удалось загрузить шаблон: {template_file}")
            continue

        tw, th = template.shape[::-1]
        res = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        matches = list(zip(*loc[::-1]))
        if matches:
            top_left = matches[0]
            bottom_right = (top_left[0] + tw, top_left[1] + th)
            field_coords = (*top_left, *bottom_right)
            if validate_area(field_coords, board_height_count, board_width_count, cell_size=CELL_SIZE):
                print(f"[SUCCESS] Поле найдено по шаблону '{template_file}': {field_coords}")
                overlay_game_area(field_coords, CELL_SIZE)
                return field_coords

    print("[INFO] Поле Сапёра не найдено")
    return None

# ==============================
# === Работа с игровым полем ==
# ==============================
def analyze_board_worker(area, cell_size, total_mines):
    """Поток для непрерывного анализа доски"""
    global current_board, stop_solver
    last_board = None

    while not stop_solver:
        if pause_solver:
            time.sleep(1)
            continue

        try:
            # Получаем текущее состояние доски
            correct_board = True

            with board_lock:
                if last_board:
                    board = analyze_board(area, cell_size)
                    for r in range(len(board)):
                        for c in range(len(board[0])):
                            old_val = last_board[r][c]
                            new_val = board[r][c]
                            if new_val == '?':
                                correct_board = False
                                break
                        if not correct_board:
                            break
                else:
                    board = analyze_board(area, cell_size)
            if not correct_board:
                print(f"[INFO] Не все ячейки распознаны, пропускаем кадр")
                continue
            current_board = board


            # Определяем изменения
            if last_board and board != last_board:
                changes = find_board_changes(last_board, board)
                if changes:
                    board_changes.put(changes)

            last_board = [row[:] for row in board]  # Глубокая копия
            time.sleep(0.2)

        except Exception as e:
            print(f"[ERROR] В потоке анализа: {str(e)}")
            time.sleep(1)

def find_board_changes(old_board, new_board):
    """Находит изменения между двумя состояниями доски"""
    changes = defaultdict(list)

    for row in range(len(old_board)):
        for col in range(len(old_board[0])):
            old_val = old_board[row][col]
            new_val = new_board[row][col]

            if old_val != new_val:
                if new_val == '?':
                    changes[(row, col)] = (old_val, old_val)
                changes[(row, col)] = (old_val, new_val)

    return dict(changes)


def analyze_board(area, cell_size):
    global stop_solver
    board_image = ImageGrab.grab(bbox=area)
    board_width = area[2] - area[0]
    board_height = area[3] - area[1]

    cols = board_width // cell_size
    rows = board_height // cell_size

    board_state = []
    for row in range(rows):
        line = []
        for col in range(cols):
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            cell_img = board_image.crop((x1, y1, x2, y2))
            cell_type = cell_template_to_type(cell_img)
            line.append(cell_type)
            if stop_solver:
                print("[INFO] Прерывание по ESC.")
                break
        board_state.append(line)

    return board_state

def get_neighbors(board, row, col):
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < len(board) and 0 <= c < len(board[0]):
                neighbors.append((r, c))
    return neighbors

def is_game_over(board, total_mines=0):
    closed_cells = 0
    flagged_cells = 0

    for row in board:
        for cell in row:
            if cell == 'X':
                closed_cells += 1
            elif cell == 'F':
                flagged_cells += 1

    # Сценарий 1: Все ячейки открыты
    if closed_cells == 0:
        print("[INFO] Все ячейки открыты. Игра завершена.")
        return True

    # Сценарий 2: Все мины помечены и больше нет закрытых
    if flagged_cells == total_mines and closed_cells == 0:
        print("[INFO] Все мины найдены и отмечены. Игра завершена.")
        return True

    # Сценарий 2: Все мины помечены, остальные — можно открыть
    if flagged_cells == total_mines and closed_cells > 0:
        print(f"[ACTION] Все мины найдены! Осталось {closed_cells} безопасных ячеек для открытия")
        return False  # Не завершаем игру — решатель должен открыть их

    return False

    return False

# ==============================
# === Логика решения игры ====
# ==============================
def get_cell_real_count_mines(board, total_mines=0):
    closed_cells = []
    flagged_cells = 0

    for row_idx, row in enumerate(board):
        for col_idx, cell in enumerate(row):
            if cell == 'X':
                closed_cells.append((row_idx, col_idx))
            elif cell == 'F':
                flagged_cells += 1

    return (total_mines - flagged_cells)

def find_cells_with_known_mines(board):
    mandatory_mines = []
    safe_cells = []
    safe_clocked = []
    error_flagged = False
    is_cloacked = False
    number_cells = []
    safe_mana = []


    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col].isdigit() and board[row][col] != '0':
                count = int(board[row][col])
                neighbors = get_neighbors(board, row, col)
                none = [n for n in neighbors if board[n[0]][n[1]] == '?']
                mana = [n for n in neighbors if board[n[0]][n[1]] == 'M']
                cloacked = [n for n in neighbors if board[n[0]][n[1]] == 'C']
                if cloacked:
                    is_cloacked = True
                flagged = [n for n in neighbors if board[n[0]][n[1]] == 'F']
                unknown = [n for n in neighbors if board[n[0]][n[1]] == 'X']
                remaining = count - len(flagged)
                number_cells.append({
                    'pos': (row, col),
                    'count': count,
                    'flagged': flagged,
                    'unknown': unknown,
                    'remaining': remaining,
                    'cloacked': cloacked,
                    'mana': mana,
                    'none': none
                })

    # === ОДИНОЧНАЯ ОБРАБОТКА ЧИСЕЛ ===

    for data in number_cells:
        if data['none']:
            continue
        move = False
        unknown = data['unknown']
        cloacked = data['cloacked']
        flagged = data['flagged']
        count = data['count']
        mana = data['mana']
        # print(f"[SOLO] Анализируем ячейку {data['pos']}")
        # print(f"   └── Неизвестные: {unknown}")
        # print(f"   └── Нужно мин: {count}")
        # print(f"   └── Помечено мин: {flagged}")

        # if is_cloacked and not cloacked:
        #     continue

        if count < len(flagged):
            for r, c in flagged:
                mandatory_mines.append((r, c))
            print(f"[SOLO] Случай 0: поставлены лишние флаги. Снимаем их {flagged}")
            error_flagged = True
            continue
        if error_flagged:
            break

        # Случай 1: Все флаги уже поставлены → открываем остальные
        if count == len(flagged):
            for r, c in cloacked:
                safe_clocked.append((r, c))

            for r, c in mana:
                safe_mana.append((r, c))

            for r, c in unknown:
                safe_cells.append((r, c))
                # move = True

        # Случай 2: Осталось ровно столько мин → ставим флаги
        elif count == len(flagged) + len(unknown) + len(cloacked) + len(mana):
            for r, c in cloacked:
                mandatory_mines.append((r, c))
                # move = True

            for r, c in unknown:
                mandatory_mines.append((r, c))
                # move = True

    return list(set(mandatory_mines)), list(set(safe_cells)), list(set(safe_clocked)), list(set(safe_mana))

def find_safe_cells(board, total_mines=0):
    mandatory_mines = []
    safe_clocked = []
    safe_cells = []
    safe_mana = []
    end = False

    number_cells = []

    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col].isdigit() and board[row][col] != '0':
                count = int(board[row][col])
                neighbors = get_neighbors(board, row, col)
                none = [n for n in neighbors if board[n[0]][n[1]] == '?']
                mana = [n for n in neighbors if board[n[0]][n[1]] == 'M']
                cloacked = [n for n in neighbors if board[n[0]][n[1]] == 'C']
                flagged = [n for n in neighbors if board[n[0]][n[1]] == 'F']
                unknown = [n for n in neighbors if board[n[0]][n[1]] == 'X' or board[n[0]][n[1]] == 'C' or board[n[0]][n[1]] == 'M']
                remaining = count - len(flagged)
                number_cells.append({
                    'pos': (row, col),
                    'count': count,
                    'flagged': flagged,
                    'unknown': unknown,
                    'remaining': remaining,
                    'cloacked': cloacked,
                    'mana': mana,
                    'none': none
                })


    # Перебираем все пары чисел
    for i in range(len(number_cells)):
        data1 = number_cells[i]
        if data1['none']:
            continue
        for j in range(i + 1, len(number_cells)):
            data2 = number_cells[j]
            if data2['none']:
                continue
            common = set(data1['unknown']) & set(data2['unknown'])
            all_unknown = set(data1['unknown']) | set(data2['unknown'])
            if not common:
                continue

            # print(f"[GROUP] Сравниваем {data1['pos']} и {data2['pos']}")
            # print(f"   └── Общие неизвестные: {common}")
            # print(f"   └── Всё неизвестные: {all_unknown}")
            # print(f"   └── Нужно мин: {data1['remaining']}, {data2['remaining']}")


            # === Случай 1: все ячейки кроме общих — безопасны  ===
            if (data1['remaining'] + data2['remaining'] == len(common) and
                    (len(data1['unknown']) == len(common) or len(data2['unknown']) == len(common)) and
                    len(all_unknown) > len(common)):
                # print("   ✅ Все ячейки кроме общих — безопасны")
                safe_cells.extend(all_unknown - common)

            # === Случай 2: число мин меньше количества общих ячеек ===
            elif ((len(data1['unknown']) == len(common) or len(data2['unknown']) == len(common)) and
                  (data1['remaining'] + data2['remaining'] == len(all_unknown)) and len(all_unknown) > len(common)):
                mandatory_mines.extend(all_unknown - common)
                # print(f"   ✅ Нужно пометить {all_unknown - common} как мины")

            else:
                real_count_mines = get_cell_real_count_mines(board, total_mines)
                print(f"   └── количество мин: {real_count_mines}")
                if len(common) == real_count_mines and data1['remaining'] == data2['remaining'] == real_count_mines:
                    mandatory_mines.extend(common)
                    # print(f"   ✅ Ячейка {common} — единственная возможная мина")
                    end = True

            if end:
                break
        if end:
            break

    return list(set(mandatory_mines)), list(set(safe_cells)), list(set(safe_clocked)), list(set(safe_mana))

def solve_minesweeper(area, cell_size=20, total_mines=10):
    global stop_solver
    global pause_solver
    stop_solver = False
    start_time = time.time()  # ← запоминаем время старта

    # Запускаем поток для анализа доски
    analyzer_thread = threading.Thread(
        target=analyze_board_worker,
        args=(area, cell_size, total_mines),
        daemon=True
    )
    analyzer_thread.start()

    # Запускаем прослушивание Esc в отдельном потоке
    escape_thread = threading.Thread(target=listen_for_escape, daemon=True)
    escape_thread.start()

    try:
        update_time = 2
        changes = {(0, 0)}
        last_board = []
        max_retries = 10
        retry_count = 0
        board = None
        
        while not stop_solver:
            time.sleep(0.1)
            if pause_solver:
                print("[INFO] Решатель приостановлен. Ожидание...")
                while pause_solver:
                    time.sleep(0.5)
                print("[INFO] Продолжаю выполнение...")
                continue

            # Получаем текущее состояние доски
            with board_lock:
                if board == current_board:
                    move_mouse_relative()
                    retry_count += 1
                    time.sleep(0.25)
                    if retry_count >= max_retries:
                        print(f"[ERROR] Превышено количество попыток ({max_retries}) при обнаружении изменений. Прерываю выполнение...")
                        if find_and_click_image("game_over/win.png"):
                            x, y = find_and_click_image("game_over/win.png")
                            click_to(x, y)
                            time.sleep(0.1)
                            click_to(x, y)
                            press_f9()
                            end_time = time.time()
                            elapsed = end_time - start_time
                            print(f"{SUCCESS}[SUCCESS] Игра решена за {int(elapsed)} секунд{RESET}")
                            break
                        elif find_and_click_image("game_over/failed.png"):
                            end_time = time.time()
                            elapsed = end_time - start_time
                            print(f"{ERROR}[ERROR] Игра проиграна за {int(elapsed)} секунд{RESET}")
                            break
                        else:
                            retry_count = 0
                            pause_solver = True
                            show_unknown_mine_message(True)
                    # print("[INFO] Доска не изменилась. Пропускаем кадр...")
                    continue
                else:
                    retry_count = 0



                board = [row[:] for row in current_board] if current_board else None

            if not board:
                time.sleep(0.5)
                continue

            if is_game_over(board, total_mines):
                time.sleep(3)
                if find_and_click_image("game_over/win.png"):
                    x, y = find_and_click_image("game_over/win.png")
                    click_to(x, y)
                    time.sleep(0.1)
                    click_to(x, y)
                    press_f9()
                    end_time = time.time()
                    elapsed = end_time - start_time
                    print(f"{SUCCESS}[SUCCESS] Игра решена за {int(elapsed)} секунд{RESET}")
                    break

            retry_count = 0

            last_board = [row[:] for row in board]  # Глубокая копия

            # Проверяем изменения
            try:
                changes = board_changes.get_nowait()
                print("[INFO] Обнаружены изменения на доске:")
                for (r, c), (old, new) in changes.items():
                    print(f"  [{r}][{c}]: {old} → {new}")
            except queue.Empty:
                pass



            print_colored_board(board)

            made_move = False

            # === Анализ обязательных мин ===
            if not made_move:
                mandatory_mines, safe_cells, safe_clocked, safe_mana = find_cells_with_known_mines(board)
                print(f"[ACTION][SOLO] Найдены обязательные мины: {mandatory_mines}")

                if safe_clocked:
                    print("[ACTION][SOLO] Найдены безопасные ячейки с часиками:")
                    for r, c in safe_clocked:
                        print(f"Часы на [{r}][{c}]")
                        click(area, r, c, right_click=False, CELL_SIZE=cell_size)
                        # time.sleep(0.1)
                        # break
                    made_move = True

                if safe_mana:
                    print("[ACTION][SOLO] Найдены безопасные ячейки с маной:")
                    for r, c in safe_mana:
                        print(f"Мана на [{r}][{c}]")
                        click(area, r, c, right_click=False, CELL_SIZE=cell_size)
                        # time.sleep(0.1)
                        # break
                    made_move = True

                if mandatory_mines:
                    print("[ACTION][SOLO] Найдены обязательные мины:")
                    for r, c in mandatory_mines:
                        print(f"Флаг на [{r}][{c}]")
                        click(area, r, c, right_click=True, CELL_SIZE=cell_size)
                    time.sleep(0.8)
                    made_move = True

                if safe_cells:
                    print("[ACTION][SOLO] Найдены безопасные ячейки:")
                    for r, c in safe_cells:
                        print(f"Открытие [{r}][{c}]")
                        click(area, r, c, right_click=False, CELL_SIZE=cell_size)
                        # time.sleep(0.1)
                        # break
                    made_move = True


            if not made_move:
                mandatory_mines, safe_cells, safe_clocked, safe_mana = find_safe_cells(board, total_mines)
                print(f"[ACTION][GROUP] Найдены обязательные мины: {mandatory_mines}")
                if safe_clocked:
                    ptint("[ACTION][GROUP] Найдены безопасные ячейки с часиками:")
                    for r, c in safe_clocked:
                        print(f"Часы на [{r}][{c}]")
                        click(area, r, c, right_click=False, CELL_SIZE=cell_size)
                        # time.sleep(0.1)
                        # break
                    made_move = True

                if safe_mana:
                    print("[ACTION][GROUP] Найдены безопасные ячейки с маной:")
                    for r, c in safe_mana:
                        print(f"Мана на [{r}][{c}]")
                        click(area, r, c, right_click=False, CELL_SIZE=cell_size)
                        # time.sleep(0.1)
                        # break
                    made_move = True

                if mandatory_mines:
                    print("[ACTION][GROUP] Найдены обязательные мины:")
                    for r, c in mandatory_mines:
                        print(f"Флаг на [{r}][{c}]")
                        click(area, r, c, right_click=True, CELL_SIZE=cell_size)
                    time.sleep(0.8)
                    made_move = True

                if safe_cells:
                    print("[ACTION][GROUP] Найдены безопасные ячейки:")
                    for r, c in safe_cells:
                        print(f"Открытие [{r}][{c}]")
                        click(area, r, c, right_click=False, CELL_SIZE=cell_size)
                        # time.sleep(0.1)
                        # break
                    made_move = True

            if not made_move:
                print("[INFO] Нет явных ходов. Проверяем, остались ли только безопасные ячейки...")

                # Если все мины найдены, открываем все оставшиеся 'X'
                closed_cells = []
                flagged_cells = 0

                for row_idx, row in enumerate(board):
                    for col_idx, cell in enumerate(row):
                        if cell == 'X':
                            closed_cells.append((row_idx, col_idx))
                        elif cell == 'F':
                            flagged_cells += 1

                if flagged_cells == total_mines and len(closed_cells) > 0:
                    print(f"[ACTION] Все мины найдены. Открываем {len(closed_cells)} ячеек")
                    for r, c in closed_cells:
                        print(f"[ACTION] Открываем [{r}][{c}]")
                        click(area, r, c, right_click=False, CELL_SIZE=cell_size)
                        # time.sleep(0.1)
                        made_move = True
                        # break  # После одного клика обновляем доску

            if not made_move:
                print(f"{WARNING}[INFO] Нет явных ходов. Предоставляется выбор пользователю...{RESET}")
                pause_solver = True
                show_unknown_mine_message(True)


            if stop_solver:
                print("[INFO] Прерывание по ESC.")
                break



        if changes == {}:
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"{WARNING}[WARNING] Игра завершена за {int(elapsed)} секунд{RESET}")


    except KeyboardInterrupt:
        print("Решатель остановлен пользователем.")
    finally:
        stop_solver = True
        analyzer_thread.join(timeout=1)
        escape_thread.join(timeout=1)
        print("[INFO] Все потоки остановлены")


pyautogui.PAUSE = 0.01  # минимальная пауза между действиями

def press_f9(delay=0.1):
    """
    Имитирует нажатие и отпускание клавиши F9

    :param delay: задержка между нажатием и отжатием (в секундах)
    """
    print("[INFO] Нажимаю F9...")
    pyautogui.keyDown('f9')   # Зажать F9
    time.sleep(delay)
    pyautogui.keyUp('f9')     # Отпустить F9
    print("[INFO] F9 отжата")

def move_mouse_relative(dx=40, dy=40):
    x, y = pyautogui.position()
    pyautogui.move(dx, dy)
    print(f"Перемещаем курсор на {dx}x{dy} = {x + dx}x{y + dy}")
    return

def click_to(x, y):
    pyautogui.moveTo(x, y)
    pyautogui.mouseDown(button='left')
    time.sleep(0.005)
    pyautogui.mouseUp(button='left')


def click(area, row, col, right_click=False, CELL_SIZE=20):
    x_center = area[0] + col * CELL_SIZE + CELL_SIZE // 2
    y_center = area[1] + row * CELL_SIZE + CELL_SIZE // 2
    pyautogui.moveTo(x_center, y_center)
    if right_click:
        # Правый клик через удержание
        pyautogui.mouseDown(button='right')
        time.sleep(0.005)
        pyautogui.mouseUp(button='right')
    else:
        # Левый клик
        pyautogui.mouseDown(button='left')
        time.sleep(0.005)
        pyautogui.mouseUp(button='left')

    # print(f"[DEBUG] {'Правый' if right_click else 'Левый'} клик по [{row}][{col}]")

# ==============================
# === Диалоги и взаимодействие
# ==============================
def select_cell_area():
    print("Выберите ячейку...")
    AreaSelectorApp(analyze_cell_color, is_cell=True).run()

def select_game_area():
    print("Выберите область игрового поля...")
    AreaSelectorApp(start_game).run()

def start_game(area, total_mines, CELL_SIZE=36, *args):
    try:
        print("Область выбрана для игры:", area)
        # Автоопределение размера ячейки
        board_width = area[2] - area[0]
        board_height = area[3] - area[1]
        board_width_count = int(width_entry.get())
        board_height_count = int(height_entry.get())
        # total_mines = int(mines_entry.get())
        # CELL_SIZE = min(board_width // board_width_count, board_height // board_height_count)
        # print(f"Определённый размер ячейки: {CELL_SIZE}")

        print("Запускаю решатель...")
        def start_solver():
            solve_minesweeper(area, cell_size=CELL_SIZE, total_mines=total_mines)

        # Запускаем решатель в отдельном потоке
        thread = threading.Thread(target=start_solver, daemon=True)
        thread.start()
    except ValueError:
        print(f"{ERROR}Ошибка ввода. Пожалуйста, введите корректные значения.{RESET}")

def start_game_auto(board_width_count = 9, board_height_count = 9, total_mines = 10):
    try:
        # board_width_count = int(width_entry.get())
        # board_height_count = int(height_entry.get())
        CELL_SIZE = detect_cell_size()
        area = find_minesweeper_field(board_width_count, board_height_count, CELL_SIZE)
        print(f"{SUCCESS}Область выбрана для игры:{area}{RESET}")
        start_game(area, total_mines, CELL_SIZE)

    except Exception as e:
        print(f"{ERROR}[ERROR] Ошибка запуска игры: {e}{RESET}")

def start_game_auto_default():
    try:
        board_width_count = int(width_entry.get())
        board_height_count = int(height_entry.get())
        total_mines = int(mines_entry.get())
        CELL_SIZE = detect_cell_size()
        area = find_minesweeper_field(board_width_count, board_height_count, CELL_SIZE)
        print(f"{SUCCESS}Область выбрана для игры:{area}{RESET}")
        start_game(area, total_mines, CELL_SIZE)

    except Exception as e:
        print(f"{ERROR}[ERROR] Ошибка запуска игры: {e}{RESET}")

def analyze_cell_color(area, *args):
    screenshot = ImageGrab.grab(bbox=area)
    cell_img = screenshot.crop((0, 0, area[2]-area[0], area[3]-area[1]))
    img = cell_img.convert('RGB')
    img_array = np.array(img)

    h, w, _ = img_array.shape
    pad_w, pad_h = w // 10, h // 10
    center = img_array[pad_h:h - pad_h, pad_w:w - pad_w]

    avg_color = np.mean(center, axis=(0, 1)).astype(int)  # RGB
    print(f"[Цвет ячейки] RGB{tuple(avg_color)}")
    update_gui_color(avg_color)


# === Обновление цвета в интерфейсе ===
def update_gui_color(rgb):
    hex_color = '#%02x%02x%02x' % tuple(rgb)
    color_label.config(bg=hex_color)
    color_value.config(text=f"RGB{tuple(rgb)}")

def show_unknown_color_dialog(cell_img):
    dialog = tk.Toplevel(root)
    dialog.title("Неизвестная ячейка")
    dialog.geometry("400x600")
    dialog.transient(root)
    dialog.grab_set()

    tk.Label(dialog, text="Обнаружена неопределённая ячейка:", font=("Arial", 12)).pack(pady=5)

    # Конвертируем в PhotoImage для отображения в Tkinter
    img_tk = ImageTk.PhotoImage(cell_img.resize((60, 60)))
    panel = tk.Label(dialog, image=img_tk)
    panel.image = img_tk  # ← ВАЖНО! Сохраняем ссылку на изображение
    panel.pack(pady=5)

    tk.Label(dialog, text="Выберите тип ячейки или закройте окно").pack(pady=5)

    def resume():
        dialog.destroy()
        global pause_solver
        pause_solver = False

    def add_template(label):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("templates", f"{label}_{timestamp}.png")
        cell_img.save(path)
        print(f"[ACTION] Сохранён новый шаблон: {label}_{timestamp}.png")
        load_templates()  # Обновляем шаблоны
        resume()

    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=10)

    for label in ['X', 'F', 'C', 'M', '0', '1', '2', '3', '4', '5', '6', '7', '8']:
        tk.Button(btn_frame, text=label, width=3, command=lambda l=label: add_template(l)).pack(side='left', padx=2)

    tk.Button(dialog, text="Выход", width=10, command=lambda: dialog.destroy()).pack(pady=5)
    root.wait_window(dialog)

def show_unknown_mine_message(show=True):
    if show:
        message_frame.pack(pady=10)  # Показываем сообщение
    else:
        message_frame.pack_forget()  # Скрываем сообщение

def resume_solver():
    global pause_solver
    pause_solver = False
    show_unknown_mine_message(False)  # Скрываем сообщение
    print("Решатель продолжает работу...")

def show_uncnown_mine_dialog():
    dialog = tk.Toplevel(root)
    dialog.title("Нет вычисляемых ходов")
    dialog.geometry("300x150")
    dialog.transient(root)
    dialog.grab_set()

    tk.Label(dialog, text="Нажмите 'ОК', чтобы продолжить").pack(pady=5)

    def resume():
        dialog.destroy()
        global pause_solver
        pause_solver = False  # Возобновляем работу решателя

    tk.Button(dialog, text="Ок", command=resume).pack(pady=5)

    root.wait_window(dialog)

# ==============================
# === Системные функции ========
# ==============================

def listen_for_escape():
    global stop_solver
    keyboard.wait('esc')
    stop_solver = True
    print("\n[INFO] Нажата клавиша ESC. Завершение работы...")



# ==============================
# === Точка входа ============
# ==============================

if __name__ == "__main__":

    load_templates()
    root = tk.Tk()
    root.title("Сапёр: Цветовой анализатор")
    root.geometry("400x800")
    root.resizable(False, False)

    tk.Label(root, text="Сапёр: Анализ цветов", font=("Arial", 16)).pack(pady=10)

    # Поля ввода настроек
    settings_frame = tk.Frame(root)
    settings_frame.pack(pady=10)

    tk.Label(settings_frame, text="Ширина поля:", width=15, anchor='w').grid(row=0, column=0, padx=5, pady=5)
    global width_entry
    width_entry = tk.Entry(settings_frame, width=10)
    width_entry.grid(row=0, column=1, padx=5, pady=5)
    width_entry.insert(0, "9")

    tk.Label(settings_frame, text="Высота поля:", width=15, anchor='w').grid(row=1, column=0, padx=5, pady=5)
    global height_entry
    height_entry = tk.Entry(settings_frame, width=10)
    height_entry.grid(row=1, column=1, padx=5, pady=5)
    height_entry.insert(0, "9")

    tk.Label(settings_frame, text="Количество мин:", width=15, anchor='w').grid(row=2, column=0, padx=5, pady=5)
    global mines_entry
    mines_entry = tk.Entry(settings_frame, width=10)
    mines_entry.grid(row=2, column=1, padx=5, pady=5)
    mines_entry.insert(0, "10")

    # Кнопки
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    tk.Button(btn_frame, text="Определить цвет ячейки", width=25, command=select_cell_area).grid(
        row=0, column=0, padx=5, pady=5)

    tk.Button(btn_frame, text="Играть", width=25, command=select_game_area).grid(
        row=1, column=0, padx=5, pady=5)

    tk.Button(btn_frame, text="Автопоиск поля", width=25, command=start_game_auto_default).grid(
        row=2, column=0, padx=5, pady=5)

    start_game_auto_9x9 = partial(start_game_auto, 9, 9, 10)
    tk.Button(btn_frame, text="9x9", width=25, command=start_game_auto_9x9).grid(
        row=3, column=0, padx=5, pady=5)

    start_game_auto_12x11 = partial(start_game_auto, 12, 11, 19)
    tk.Button(btn_frame, text="12x11", width=25, command=start_game_auto_12x11).grid(
        row=4, column=0, padx=5, pady=5)

    start_game_auto_15x13 = partial(start_game_auto, 15, 13, 32)
    tk.Button(btn_frame, text="15x13", width=25, command=start_game_auto_15x13).grid(
        row=5, column=0, padx=5, pady=5)

    start_game_auto_18x14 = partial(start_game_auto, 18, 14, 47)
    tk.Button(btn_frame, text="18x14", width=25, command=start_game_auto_18x14).grid(
        row=6, column=0, padx=5, pady=5)

    start_game_auto_20x16 = partial(start_game_auto, 20, 16, 66)
    tk.Button(btn_frame, text="20x16", width=25, command=start_game_auto_20x16).grid(
        row=7, column=0, padx=5, pady=5)

    # ==== Сообщение ====
    message_frame = tk.Frame(root)
    tk.Label(message_frame, text="Нет вычисляемых ходов", fg="red", font=("Arial", 12)).pack()
    tk.Label(message_frame, text="Нажмите 'Ок', чтобы продолжить").pack(pady=5)
    tk.Button(message_frame, text="Ок", command=resume_solver).pack(pady=5)

    # Скрываем по умолчанию
    message_frame.pack_forget()

    # ==== Тестовая кнопка для вызова сообщения ====
    tk.Button(root, text="Показать сообщение", command=lambda: (
        show_unknown_mine_message(True),
        print("Решатель остановлен...")
    )).pack(pady=20)

    # Отображение цвета
    global color_label, color_value
    color_label = tk.Label(root, text="Цвет", width=20, height=3, relief="solid")
    color_label.pack(pady=10)

    color_value = tk.Label(root, text="RGB(?, ?, ?)", font=("Courier", 12))
    color_value.pack()

    root.mainloop()


