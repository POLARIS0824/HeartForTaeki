from tkinter import *
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random
import math
import numpy as np
import os
import colorsys
import cv2
from scipy.ndimage.filters import gaussian_filter

canvas_width = 600
canvas_height = 600
world_width = 0.05
world_heigth = 0.05

# 中间心的参数
points = None
fixed_point_size = 20000
fixed_scale_range = (4, 4.3)
min_scale = np.array([1.0, 1.0, 1.0]) * 0.9
max_scale = np.array([1.0, 1.0, 1.0]) * 0.9
min_heart_scale = -15
max_heart_scale = 16

# 外围随机心参数
random_point_szie = 7000
random_scale_range = (3.5, 3.9)
random_point_maxvar = 0.2

# 心算法参数
mid_point_ignore = 0.95

# 相机参数
camera_close_plane = 0.1
camera_position = np.array([0.0, -2.0, 0.0])

# 点的颜色
hue = 0.92
color_strength = 255

# 常用向量缓存
zero_scale = np.array([0.0, 0.0, 0.0])
unit_scale = np.array([1.0, 1.0, 1.0])
color_white = np.array([255, 255, 255])
axis_y = np.array([0.0, 1.0, 0.0])

# 渲染缓存
render_buffer = np.empty((canvas_width, canvas_height, 3), dtype=int)
strength_buffer = np.empty((canvas_width, canvas_height), dtype=float)

# 随机点文件缓存
points_file = "temp.txt"

# 渲染结果
total_frames = 30
output_dir = "./output"

# 格式
image_fmt = "jpg"

# 文字参数
text_display_time = 3000  # 文字显示时间从5000毫秒减少到2000毫秒
text_font_size = 80
text_content = "To Taeki"
heart_display_cycles = 3  # 心形显示循环次数


def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    string = '#'
    for i in value:
        a1 = i // 16
        a2 = i % 16
        string += digit[a1] + digit[a2]
    return string


def heart_func(x, y, z, scale):
    bscale = scale
    bscale_half = bscale / 2
    x = x * bscale - bscale_half
    y = y * bscale - bscale_half
    z = z * bscale - bscale_half
    return (x ** 2 + 9 / 4 * (y ** 2) + z ** 2 - 1) ** 3 - (x ** 2) * (z ** 3) - 9 / 200 * (y ** 2) * (z ** 3)


def lerp_vector(a, b, ratio):
    result = a.copy()
    for i in range(3):
        result[i] = a[i] + (b[i] - a[i]) * ratio
    return result


def lerp_int(a, b, ratio):
    return (int)(a + (b - a) * ratio)


def lerp_float(a, b, ratio):
    return (a + (b - a) * ratio)


def distance(point):
    return (point[0] ** 2 + point[1] ** 2 + point[2] ** 2) ** 0.5


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def inside_rand(tense):
    x = random.random()
    y = -tense * math.log(x)
    return y


# 生成中间心
def genPoints(pointCount, heartScales):
    result = np.empty((pointCount, 3))
    index = 0
    while index < pointCount:
        # 生成随机点
        x = random.random()
        y = random.random()
        z = random.random()

        # 扣掉心中间的点
        mheartValue = heart_func(x, 0.5, z, heartScales[1])
        mid_ignore = random.random()
        if mheartValue < 0 and mid_ignore < mid_point_ignore:
            continue

        heartValue = heart_func(x, y, z, heartScales[0])
        z_shrink = 0.01
        sz = z - z_shrink
        sheartValue = heart_func(x, y, sz, heartScales[1])

        # 保留在心边上的点
        if heartValue < 0 and sheartValue > 0:
            result[index] = [x - 0.5, y - 0.5, z - 0.5]

            # 向内扩散
            len = 0.7
            result[index] = result[index] * (1 - len * inside_rand(0.2))

            # 重新赋予深度
            newY = random.random() - 0.5
            rheartValue = heart_func(result[index][0] + 0.5, newY + 0.5, result[index][2] + 0.5, heartScales[0])
            if rheartValue > 0:
                continue
            result[index][1] = newY

            # 删掉肚脐眼
            dist = distance(result[index])
            if dist < 0.12:
                continue

            index = index + 1
            if index % 100 == 0:
                print("{ind} generated {per}%".format(ind=index, per=((index / pointCount) * 100)))

    return result


# 生成随机心
def genRandPoints(pointCount, heartScales, maxVar, ratio):
    result = np.empty((pointCount, 3))
    index = 0
    while index < pointCount:
        x = random.random()
        y = random.random()
        z = random.random()
        mheartValue = heart_func(x, 0.5, z, heartScales[1])
        mid_ignore = random.random()
        if mheartValue < 0 and mid_ignore < mid_point_ignore:
            continue

        heartValue = heart_func(x, y, z, heartScales[0])
        sheartValue = heart_func(x, y, z, heartScales[1])

        if heartValue < 0 and sheartValue > 0:
            result[index] = [x - 0.5, y - 0.5, z - 0.5]
            dist = distance(result[index])
            if dist < 0.12:
                continue

            len = 0.7
            result[index] = result[index] * (1 - len * inside_rand(0.2))
            index = index + 1

    for i in range(pointCount):
        var = maxVar * ratio
        randScale = 1 + random.normalvariate(0, var)
        result[i] = result[i] * randScale

    return result


# 世界坐标到相机本地坐标
def world_2_cameraLocalSapce(world_point):
    new_point = world_point.copy()
    new_point[1] = new_point[1] + camera_position[1]
    return new_point


# 相机本地坐标到相机空间坐标
def cameraLocal_2_cameraSpace(cameraLocalPoint):
    depth = distance(cameraLocalPoint)
    cx = cameraLocalPoint[0] * (camera_close_plane / cameraLocalPoint[1])
    cz = -cameraLocalPoint[2] * (cx / cameraLocalPoint[0])
    cameraLocalPoint[0] = cx
    cameraLocalPoint[1] = cz
    return cameraLocalPoint, depth


# 相机空间坐标到屏幕坐标
def camerSpace_2_screenSpace(cameraSpace):
    x = cameraSpace[0]
    y = cameraSpace[1]

    # convert to view space
    centerx = canvas_width / 2
    centery = canvas_height / 2
    ratiox = canvas_width / world_width
    ratioy = canvas_height / world_heigth

    viewx = centerx + x * ratiox
    viewy = canvas_height - (centery + y * ratioy)

    cameraSpace[0] = viewx
    cameraSpace[1] = viewy
    return cameraSpace.astype(int)


# 绘制世界坐标下的点
def draw_point(worldPoint):
    cameraLocal = world_2_cameraLocalSapce(worldPoint)
    cameraSpsace, depth = cameraLocal_2_cameraSpace(cameraLocal)
    screeSpace = camerSpace_2_screenSpace(cameraSpsace)

    draw_size = int(random.random() * 3 + 1)
    draw_on_buffer(screeSpace, depth, draw_size)


# 绘制到缓存上
def draw_on_buffer(screenPos, depth, draw_size):
    if draw_size == 0:
        return
    elif draw_size == 1:
        draw_point_on_buffer(screenPos[0], screenPos[1], color_strength, depth)
    elif draw_size == 2:
        draw_point_on_buffer(screenPos[0], screenPos[1], color_strength, depth)
        draw_point_on_buffer(screenPos[0] + 1, screenPos[1] + 1, color_strength, depth)
    elif draw_size == 3:
        draw_point_on_buffer(screenPos[0], screenPos[1], color_strength, depth)
        draw_point_on_buffer(screenPos[0] + 1, screenPos[1] + 1, color_strength, depth)
        draw_point_on_buffer(screenPos[0] + 1, screenPos[1], color_strength, depth)
    elif draw_size == 4:
        draw_point_on_buffer(screenPos[0], screenPos[1], color_strength, depth)
        draw_point_on_buffer(screenPos[0] + 1, screenPos[1], color_strength, depth)
        draw_point_on_buffer(screenPos[0], screenPos[1] + 1, color_strength, depth)
        draw_point_on_buffer(screenPos[0] + 1, screenPos[1] + 1, color_strength, depth)


# 根据色调和颜色强度获取颜色
def get_color(strength):
    result = None
    if strength >= 1:
        result = colorsys.hsv_to_rgb(hue, 2 - strength, 1)
    else:
        result = colorsys.hsv_to_rgb(hue, 1, strength)
    r = min(result[0] * 256, 255)
    g = min(result[1] * 256, 255)
    b = min(result[2] * 256, 255)
    return np.array((r, g, b), dtype=int)


# 可以根据深度做一些好玩的
def draw_point_on_buffer(x, y, color, depth):
    if x < 0 or x >= canvas_width or y < 0 or y >= canvas_height:
        return

    # 混合
    strength = float(color) / 255
    strength_buffer[x, y] = strength_buffer[x, y] + strength


# 绘制缓存
def draw_buffer_on_canvas(output=None):
    render_buffer.fill(0)
    for i in range(render_buffer.shape[0]):
        for j in range(render_buffer.shape[1]):
            render_buffer[i, j] = get_color(strength_buffer[i, j])
    im = Image.fromarray(np.uint8(render_buffer))
    im = im.rotate(-90)
    if output is None:
        plt.imshow(im)
        plt.show()
    else:
        im.save(output)


def paint_heart(ratio, randratio, outputFile=None):
    global strength_buffer
    global render_buffer
    global points

    # 清空缓存
    strength_buffer.fill(0)

    for i in range(fixed_point_size):
        # 缩放
        point = points[i] * lerp_vector(min_scale, max_scale, ratio)

        # 球型场
        dist = distance(point)
        radius = 0.4
        sphere_scale = radius / dist
        point = point * lerp_float(0.9, sphere_scale, ratio * 0.3)

        # 绘制
        draw_point(point)

    # 生成一组随机点
    randPoints = genRandPoints(random_point_szie, random_scale_range, random_point_maxvar, randratio)
    for i in range(random_point_szie):
        # 绘制
        draw_point(randPoints[i])

    # 高斯模糊
    for i in range(1):
        strength_buffer = gaussian_filter(strength_buffer, sigma=0.8)

    # 绘制缓存
    draw_buffer_on_canvas(outputFile)


# 生成文字图像
def generate_text_image():
    # 创建空白图像，与心形图案大小相同
    img = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 尝试加载现代字体，如果找不到则使用默认字体
    try:
        # 尝试几种现代字体
        font_options = ["Roboto", "Montserrat", "Open Sans", "Segoe UI", "Calibri", "Lato", "Raleway", "Arial Nova"]
        font = None

        for font_name in font_options:
            try:
                font = ImageFont.truetype(f"{font_name}.ttf", text_font_size)
                break
            except:
                continue

        if font is None:
            # 如果没有找到现代字体，尝试系统常见字体
            system_fonts = ["arial.ttf", "segoeui.ttf", "calibri.ttf", "meiryo.ttc", "verdana.ttf"]
            for sys_font in system_fonts:
                try:
                    font = ImageFont.truetype(sys_font, text_font_size)
                    break
                except:
                    continue

        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # 获取文本大小以居中绘制
    try:
        text_bbox = draw.textbbox((0, 0), text_content, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except:
        # 旧版PIL兼容
        text_width, text_height = draw.textsize(text_content, font=font)

    # 计算居中位置
    position = ((canvas_width - text_width) // 2, (canvas_height - text_height) // 2)

    # 使用天蓝色绘制文字
    text_hue = 0.55  # 天蓝色的HSV色相值
    text_saturation = 0.85  # 饱和度
    text_value = 1.0  # 亮度
    color_rgb = colorsys.hsv_to_rgb(text_hue, text_saturation, text_value)
    r = int(color_rgb[0] * 255)
    g = int(color_rgb[1] * 255)
    b = int(color_rgb[2] * 255)

    # 添加艺术效果 - 光晕和发光效果
    # 先绘制几层模糊的文字作为光晕
    for i in range(5, 0, -1):
        blur_position = (position[0]-i, position[1]-i)
        draw.text(blur_position, text_content, fill=(r//4, g//4, b//4), font=font)
        blur_position = (position[0]+i, position[1]+i)
        draw.text(blur_position, text_content, fill=(r//4, g//4, b//4), font=font)

    # 绘制主文字
    draw.text(position, text_content, fill=(r, g, b), font=font)

    # 保存图像
    text_path = os.path.join(output_dir, "text.jpg")
    img.save(text_path)
    return text_path


def show_images():
    img = None
    text_img_path = None

    while True:
        # 显示心形动画，多循环几次
        for _ in range(heart_display_cycles):  # 增加心形显示的循环次数
            for i in range(total_frames):
                save_name = "{name}.{fmt}".format(name=i, fmt=image_fmt)
                save_path = os.path.join(output_dir, save_name)
                img = cv2.imread(save_path, cv2.IMREAD_ANYCOLOR)
                cv2.imshow("Img", img)
                cv2.waitKey(45)  # 将等待时间从60毫秒减少到45毫秒，加快切换速度

        # 生成并显示文字，仅在第一次循环时生成
        if text_img_path is None:
            text_img_path = generate_text_image()

        # 显示文字
        text_img = cv2.imread(text_img_path, cv2.IMREAD_ANYCOLOR)
        cv2.imshow("Img", text_img)
        cv2.waitKey(text_display_time)  # 文字显示指定时间


def gen_images():
    global points

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # 尝试加载或生成中间心
    if not os.path.exists(points_file):
        print("未发现缓存点，重新生成中")
        points = genPoints(fixed_point_size, fixed_scale_range)
        np.savetxt(points_file, points)
    else:
        print("发现缓存文件，跳过生成")
        points = np.loadtxt(points_file)

    for i in range(total_frames):
        print("正在处理图片... ", i)
        frame_ratio = float(i) / (total_frames - 1)
        frame_ratio = frame_ratio ** 2
        ratio = math.sin(frame_ratio * math.pi) * 0.743144
        randratio = math.sin(frame_ratio * math.pi * 2 + total_frames / 2)
        save_name = "{name}.{fmt}".format(name=i, fmt=image_fmt)
        save_path = os.path.join(output_dir, save_name)
        paint_heart(ratio, randratio, save_path)
        print("图片已保存至", save_path)


if __name__ == "__main__":
    gen_images()
    while True:
        show_images()