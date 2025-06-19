import tkinter as tk
from PIL import ImageGrab, Image, ImageDraw
import numpy as np
import cv2
import time
import pyautogui
import random
import keyboard
import threading


# === Функция для выбора области экрана ===
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

def highlight_board(area, cell_size):
    # Делаем скриншот игрового поля
    board_image = ImageGrab.grab(bbox=area)
    board_image = board_image.convert("RGBA")  # RGBA для прозрачности

    draw = ImageDraw.Draw(board_image)

    # Цвета рамок
    COLORS = {
        'closed': (0, 0, 0),      # Чёрный
        'flag': (255, 0, 0),      # Красный
        '0': (255, 255, 255),     # Белый
        '1': (255, 255, 255),
        '2': (255, 255, 255),
        '3': (255, 255, 255),
        '4': (255, 255, 255),
        '5': (255, 255, 255),
        '6': (255, 255, 255),
        '7': (255, 255, 255),
        '8': (255, 255, 255),
        '?': (255, 255, 0),       # Жёлтый для неопределённых
    }

    # Размеры доски
    board_width = area[2] - area[0]
    board_height = area[3] - area[1]

    cols = board_width // cell_size
    rows = board_height // cell_size

    # Анализируем каждую ячейку и рисуем рамку
    for row in range(rows):
        for col in range(cols):
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            cell_img = board_image.crop((x1, y1, x2, y2))
            cell_type = cell_color_to_type(cell_img)

            color = COLORS.get(cell_type, (255, 255, 0))  # Жёлтый по умолчанию

            # Рисуем рамку (толщина 2 пикселя)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    # Показываем результат поверх экрана
    overlay_window(board_image)

def overlay_window(image):
    root = tk.Tk()
    root.overrideredirect(True)  # Убираем заголовок окна
    root.geometry(f"{image.width}x{image.height}+0+0")
    root.lift()
    root.attributes("-topmost", True)
    root.attributes("-transparentcolor", "black")  # Черный фон станет прозрачным

    # Преобразуем изображение в Tkinter PhotoImage
    from PIL import ImageTk
    img = ImageTk.PhotoImage(image)

    # СОХРАНЯЕМ ССЫЛКУ НА ИЗОБРАЖЕНИЕ
    root.image = img  # ← ВАЖНО!

    label = tk.Label(root, image=img)
    label.pack()

    # Закрываем оверлей через 1 секунду
    root.after(1000, root.destroy)

    root.mainloop()

# === Определение типа ячейки по цвету ===
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
        'closed':   (186, 187, 189),
        'flag':     (175, 164, 166),
        '1':        (143, 142, 187),
        '2':        (123,159, 123),
        '3':        (195, 127, 128),
        '4':        (130, 128, 162),
        '5':        (161, 124, 126),
        '6':        (0, 0, 0),
        '7':        (133, 154, 154),
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



# === Анализ поля Сапёра ===
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


# === Получение соседних координат ===
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
            if cell == 'closed':
                closed_cells += 1
            elif cell == 'flag':
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

# === Логика решения Сапёра ===
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

                        unknown = [n for n in neighbors if board[n[0]][n[1]] == 'closed']
                        flagged = [n for n in neighbors if board[n[0]][n[1]] == 'flag']

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

            # Если явных ходов нет → делаем случайный выбор
            if not made_move:
                print("[INFO] Нет явных ходов. Ищу закрытые ячейки для случайного открытия...")

                closed_cells = []
                for row_idx, row in enumerate(board):
                    for col_idx, cell in enumerate(row):
                        if cell == 'closed':
                            closed_cells.append((row_idx, col_idx))

                if closed_cells:
                    safe_choice = random.choice(closed_cells)
                    r, c = safe_choice
                    print(f"[ACTION] Случайный выбор: открываю [{r}][{c}]")
                    click(area, r, c, right_click=False, CELL_SIZE=cell_size)
                else:
                    print("[INFO] Нет закрытых ячеек. Возможно, игра завершена.")

            if stop_solver:
                print("[INFO] Прерывание по ESC.")
                break

        time.sleep(0.1)

    except KeyboardInterrupt:
        print("Решатель остановлен пользователем.")


# === Совершить клик по ячейке ===
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

# === Вывод среднего цвета ячейки ===
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


# === Начать игру (выбрать область поля и запустить решатель) ===
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


# === Выбор области ячейки ===
def select_cell_area():
    print("Выберите ячейку...")
    AreaSelectorApp(analyze_cell_color, is_cell=True).run()


# === Выбор области игрового поля ===
def select_game_area():
    print("Выберите область игрового поля...")
    AreaSelectorApp(start_game).run()

def listen_for_escape():
    global stop_solver
    keyboard.wait('esc')
    stop_solver = True
    print("\n[INFO] Нажата клавиша ESC. Завершение работы...")

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

stop_solver = False
pause_solver = False
# === GUI приложение ===

root = tk.Tk()
root.title("Сапёр: Цветовой анализатор")
root.geometry("400x300")
root.resizable(False, False)

tk.Label(root, text="Сапёр: Анализ цветов", font=("Arial", 16)).pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=10)

tk.Label(frame, text="Ширина поля:", width=15, anchor='w').grid(row=0, column=0, padx=5, pady=5)
width_entry = tk.Entry(frame, width=10)
width_entry.grid(row=0, column=1, padx=5, pady=5)
width_entry.insert(0, "10")

tk.Label(frame, text="Высота поля:", width=15, anchor='w').grid(row=1, column=0, padx=5, pady=5)
height_entry = tk.Entry(frame, width=10)
height_entry.grid(row=1, column=1, padx=5, pady=5)
height_entry.insert(0, "10")

tk.Label(frame, text="Количество мин:", width=15, anchor='w').grid(row=2, column=0, padx=5, pady=5)
mines_entry = tk.Entry(frame, width=10)
mines_entry.grid(row=2, column=1, padx=5, pady=5)
mines_entry.insert(0, "10")

btn_select_cell = tk.Button(frame, text="Определить цвет ячейки", width=25, command=select_cell_area)
btn_select_cell.grid(row=0, column=0, padx=5, pady=5)

btn_play = tk.Button(frame, text="Играть", width=25, command=select_game_area)
btn_play.grid(row=1, column=0, padx=5, pady=5)

color_label = tk.Label(root, text="Цвет", width=20, height=3, relief="solid")
color_label.pack(pady=10)

color_value = tk.Label(root, text="RGB(?, ?, ?)", font=("Courier", 12))
color_value.pack()

root.mainloop()