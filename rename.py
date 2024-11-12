import os
from PIL import Image

# 打开图片
image = Image.open("subclass12\star\star_0.png")

# 获取图片尺寸
width, height = image.size
print("Image size:", width, "x", height)

"""
folder_path = 'subclass12/yoga'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 检查文件是否以.PNG结尾（忽略大小写）
    if filename.lower().endswith('.png'):
        # 新文件名，加上 'apple' 前缀
        new_name = 'yoga_' + filename
        # 获取完整的旧文件路径和新文件路径
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        # 重命名文件
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} to {new_name}')
"""