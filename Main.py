import tkinter as tk
from PIL import ImageGrab, Image, ImageDraw
import numpy as np
import cv2
import time
import pyautogui
import random
import keyboard
import threading
from itertools import combinations


# === Глобальные переменные ===
stop_solver = False
pause_solver = False

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
# === Распознавание цветов ====
# ==============================
def cell_color_to_type(cell_img):
    global pause_solver

    img = cell_img.convert('RGB')
    img_array = np.array(img)

    h, w, _ = img_array.shape
    pad_w, pad_h = w // 10, h // 10
    center = img_array[pad_h:h - pad_h, pad_w:w - pad_w]

    avg_color = np.mean(center, axis=(0, 1)).astype(int)  # RGB среднее

    CELL_COLORS = {
        '0':        (172, 171, 172),
        'X':        (186, 187, 189),
        'F':        (175, 164, 166),
        '1':        (143, 142, 187),
        '2':        (123,159, 123),
        '3':        (195, 127, 128),
        '4':        (130, 128, 162),
        '5':        (161, 124, 126),
        '6':        (0, 0, 0),
        '7':        (0, 0, 0),
        '8':        (133, 154, 154),
    }

    def color_distance(c1, c2):
        return sum((a - b) ** 2 for a, b in zip(c1, c2))

    min_dist = float('inf')
    closest_type = '?'

    for label, color in CELL_COLORS.items():
        dist = color_distance(avg_color, color)
        if dist < min_dist:
            min_dist = dist
            closest_type = label

    if min_dist > 3000:  # Можно настроить порог
        print(f"[DEBUG] Неопределённая ячейка | Цвет: {avg_color}, дистанция: {min_dist}")
        pause_solver = True
        root.bell()
        show_unknown_color_dialog(avg_color)

        return '?'

    return closest_type

# ==============================
# === Работа с игровым полем ==
# ==============================
def analyze_board(area, cell_size):
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
            cell_type = cell_color_to_type(cell_img)
            line.append(cell_type)
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

    return False

# ==============================
# === Логика решения игры ====
# ==============================
def find_mandatory_mines(board):
    mandatory_mines = []
    safe_cells = []

    # Собираем информацию о числах и их соседях
    number_constraints = {}
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col].isdigit() and board[row][col] != '0':
                count = int(board[row][col])
                neighbors = get_neighbors(board, row, col)
                unknown = [n for n in neighbors if board[n[0]][n[1]] == 'X']
                flagged = [n for n in neighbors if board[n[0]][n[1]] == 'F']

                # Оставшиеся неизвестные ячейки
                remaining_unknown = len(unknown) - len(flagged)

                if remaining_unknown > 0:
                    number_constraints[(row, col)] = {
                        'count': count,
                        'unknown': unknown,
                        'flagged': flagged,
                        'remaining': remaining_unknown
                    }

    # Проверяем пересечения между числами
    keys = list(number_constraints.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            cell1 = keys[i]
            cell2 = keys[j]

            # Получаем данные для двух чисел
            data1 = number_constraints[cell1]
            data2 = number_constraints[cell2]

            # Находим общие неизвестные ячейки
            common_unknown = set(data1['unknown']) & set(data2['unknown'])

            # Если количество мин совпадает с количеством общих ячеек
            if data1['remaining'] == len(common_unknown) and data2['remaining'] == len(common_unknown):
                mandatory_mines.extend(common_unknown)

            # Если общих ячеек больше, чем нужно минам
            elif len(common_unknown) > max(data1['remaining'], data2['remaining']):
                # Ячейки, которые точно безопасны
                safe_cells.extend(common_unknown)

    return mandatory_mines, safe_cells

def solve_minesweeper(area, cell_size=20, total_mines=10):
    global stop_solver
    global pause_solver
    stop_solver = False

    # Запускаем прослушивание Esc в отдельном потоке
    escape_thread = threading.Thread(target=listen_for_escape, daemon=True)
    escape_thread.start()

    try:
        while True:
            if pause_solver:
                print("[INFO] Решатель приостановлен. Ожидание...")
                while pause_solver:
                    time.sleep(0.5)
                print("[INFO] Продолжаю выполнение...")
                continue
            board = analyze_board(area, cell_size)

            # Проверяем, не закончилась ли игра
            if is_game_over(board, total_mines):
                print("[SUCCESS] Игра успешно решена!")
                break

            print("Текущее состояние доски:")
            for row in board:
                print(row)

            made_move = False

            for row_idx, row in enumerate(board):
                for col_idx, cell in enumerate(row):
                    if cell.isdigit() and cell != '0':
                        count = int(cell)
                        neighbors = get_neighbors(board, row_idx, col_idx)

                        unknown = [n for n in neighbors if board[n[0]][n[1]] == 'X']
                        flagged = [n for n in neighbors if board[n[0]][n[1]] == 'F']

                        # Случай 1: Все флаги уже поставлены → открываем остальные
                        if count == len(flagged):
                            for r, c in unknown:
                                print(f"[ACTION] Открываем [{r}][{c}]")
                                click(area, r, c, right_click=False, CELL_SIZE=cell_size)
                                time.sleep(0.1)
                                made_move = True
                                break  # Выходим из цикла и обновляем доску

                        # Случай 2: Осталось ровно столько мин → ставим флаги
                        elif count == len(flagged) + len(unknown):
                            for r, c in unknown:
                                print(f"[ACTION] Ставим флаг на [{r}][{c}]")
                                click(area, r, c, right_click=True, CELL_SIZE=cell_size)
                                time.sleep(0.1)
                                made_move = True
                                break  # Выходим из цикла и обновляем доску

                            if made_move:
                                break  # Перезапускаем анализ
                    if made_move:
                        break

            # === Анализ обязательных мин ===
            if not made_move:
                mandatory_mines, safe_cells = find_mandatory_mines(board)

                if mandatory_mines:
                    print("[ACTION] Найдены обязательные мины:")
                    for r, c in mandatory_mines:
                        print(f"Флаг на [{r}][{c}]")
                        click(area, r, c, right_click=True, CELL_SIZE=cell_size)
                    made_move = True

                elif safe_cells:
                    print("[ACTION] Найдены безопасные ячейки:")
                    for r, c in safe_cells:
                        print(f"Открытие [{r}][{c}]")
                        click(area, r, c, right_click=False, CELL_SIZE=cell_size)
                    made_move = True

            if not made_move:
                print("[INFO] Нет явных ходов. Предоставляется выбор пользователю...")
                pause_solver = True
                show_uncnown_mine_dialog()


            if stop_solver:
                print("[INFO] Прерывание по ESC.")
                break

        time.sleep(0.1)

    except KeyboardInterrupt:
        print("Решатель остановлен пользователем.")

# def analyze_groups(board):
#     rows = len(board)
#     cols = len(board[0])
#
#     # Словарь для хранения информации о группах
#     groups = {}
#
#     # Проходим по всем числам
#     for row in range(rows):
#         for col in range(cols):
#             if board[row][col].isdigit():
#                 count = int(board[row][col])
#                 neighbors = get_neighbors(board, row, col)
#
#                 # Соседние ячейки
#                 unknown = [n for n in neighbors if board[n[0]][n[1]] == 'X']
#                 flagged = [n for n in neighbors if board[n[0]][n[1]] == 'F']
#
#                 # Если число мин уже известно
#                 if count == len(flagged):
#                     # Открываем все незакрытые ячейки
#                     for r, c in unknown:
#                         yield ('open', r, c)
#                 elif count == len(flagged) + len(unknown):
#                     # Ставим флаги на все незакрытые ячейки
#                     for r, c in unknown:
#                         yield ('flag', r, c)
#                 else:
#                     # Добавляем эту группу в список для дальнейшего анализа
#                     key = (row, col)
#                     groups[key] = {
#                         'count': count,
#                         'unknown': unknown,
#                         'flagged': flagged,
#                     }
#
#     # Теперь анализируем группы вместе
#     for group_key, group_data in groups.items():
#         count = group_data['count']
#         unknown = group_data['unknown']
#         flagged = group_data['flagged']
#
#         # Проверяем пересечения с другими группами
#         for other_key, other_data in groups.items():
#             if other_key != group_key:
#                 other_count = other_data['count']
#                 other_unknown = other_data['unknown']
#                 other_flagged = other_data['flagged']
#
#                 # Найдём общих соседей
#                 common_unknown = list(set(unknown) & set(other_unknown))
#                 if common_unknown:
#                     # Вычисляем, сколько мин могут быть в общих соседях
#                     total_mines = min(count - len(flagged), other_count - len(other_flagged))
#
#                     # Если общие соседи меньше или равны количеству мин
#                     if len(common_unknown) <= total_mines:
#                         # Ставим флаги на всех общих соседях
#                         for r, c in common_unknown:
#                             yield ('flag', r, c)
#                     elif len(common_unknown) == total_mines:
#                         # Открываем все другие незакрытые ячейки
#                         for r, c in unknown:
#                             if (r, c) not in common_unknown:
#                                 yield ('open', r, c)

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

    print(f"[DEBUG] {'Правый' if right_click else 'Левый'} клик по [{row}][{col}]")

# ==============================
# === Диалоги и взаимодействие
# ==============================
def select_cell_area():
    print("Выберите ячейку...")
    AreaSelectorApp(analyze_cell_color, is_cell=True).run()

def select_game_area():
    print("Выберите область игрового поля...")
    AreaSelectorApp(start_game).run()

def start_game(area, *args):
    try:
        print("Область выбрана для игры:", area)


        # Автоопределение размера ячейки
        board_width = area[2] - area[0]
        board_height = area[3] - area[1]
        board_width_count = int(width_entry.get())
        board_height_count = int(height_entry.get())
        total_mines = int(mines_entry.get())
        CELL_SIZE = min(board_width // board_width_count, board_height // board_height_count)
        print(f"Определённый размер ячейки: {CELL_SIZE}")

        print("Запускаю решатель...")
        def start_solver():
            solve_minesweeper(area, cell_size=CELL_SIZE, total_mines=total_mines)

        # Запускаем решатель в отдельном потоке
        thread = threading.Thread(target=start_solver, daemon=True)
        thread.start()
    except ValueError:
        print("Ошибка ввода. Пожалуйста, введите корректные значения.")

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

def show_unknown_color_dialog(rgb_color):
    dialog = tk.Toplevel(root)
    dialog.title("Неизвестный цвет")
    dialog.geometry("300x150")
    dialog.transient(root)
    dialog.grab_set()

    hex_color = '#%02x%02x%02x' % tuple(rgb_color)

    tk.Label(dialog, text="Обнаружен неизвестный цвет:", font=("Arial", 12)).pack(pady=5)
    color_label = tk.Label(dialog, text=f"RGB{rgb_color}", bg=hex_color, width=20, height=2, relief="solid")
    color_label.pack(pady=5)

    tk.Label(dialog, text="Нажмите 'ОК', чтобы продолжить").pack(pady=5)

    def resume():
        dialog.destroy()
        global pause_solver
        pause_solver = False  # Возобновляем работу решателя

    tk.Button(dialog, text="Продолжить", command=resume).pack(pady=5)

    root.wait_window(dialog)

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
    root = tk.Tk()
    root.title("Сапёр: Цветовой анализатор")
    root.geometry("400x400")
    root.resizable(False, False)

    tk.Label(root, text="Сапёр: Анализ цветов", font=("Arial", 16)).pack(pady=10)

    # Поля ввода настроек
    settings_frame = tk.Frame(root)
    settings_frame.pack(pady=10)

    tk.Label(settings_frame, text="Ширина поля:", width=15, anchor='w').grid(row=0, column=0, padx=5, pady=5)
    global width_entry
    width_entry = tk.Entry(settings_frame, width=10)
    width_entry.grid(row=0, column=1, padx=5, pady=5)
    width_entry.insert(0, "10")

    tk.Label(settings_frame, text="Высота поля:", width=15, anchor='w').grid(row=1, column=0, padx=5, pady=5)
    global height_entry
    height_entry = tk.Entry(settings_frame, width=10)
    height_entry.grid(row=1, column=1, padx=5, pady=5)
    height_entry.insert(0, "10")

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

    # Отображение цвета
    global color_label, color_value
    color_label = tk.Label(root, text="Цвет", width=20, height=3, relief="solid")
    color_label.pack(pady=10)

    color_value = tk.Label(root, text="RGB(?, ?, ?)", font=("Courier", 12))
    color_value.pack()

    root.mainloop()


