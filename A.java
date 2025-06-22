# === ОБРЕЗАЕМ КРАЯ (оставляем 80% в центре)
    border_percent = 0.2  # 20%
    border_h = int(h * border_percent)
    border_w = int(w * border_percent)

    center = img_array[border_h:h - border_h, border_w:w - border_w]  # Центральная область