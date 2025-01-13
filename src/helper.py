def yolo_to_pixel(x_center, y_center, width, height, img_width, img_height):
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)

    return [x1, y1, x2, y2]


def pixel_to_yolo(x1, y1, x2, y2, img_width, img_height):
    width = (x2 - x1)
    height = (y2 - y1)

    x_center = (x1 + width / 2) / img_width
    y_center = (y1 + height / 2) / img_height
    width = width / img_width
    height = height / img_height

    return [x_center, y_center, width, height]

