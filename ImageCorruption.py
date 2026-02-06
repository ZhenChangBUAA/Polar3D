import cv2
import numpy as np
import os


def add_realistic_polar_highlight(image_path, output_path, light_dir=(0.5, -0.3, 1.0),
                                  specular_strength=0.8, glossiness=20, env_color=(20, 30, 40)):
    """
    给生物表面添加真实偏振高光（贴合曲面、带环境反光）
    :param image_path: 输入图片路径
    :param output_path: 输出图片路径
    :param light_dir: 光源方向 (x,y,z)，z正方向为光源朝画面内
    :param specular_strength: 高光强度（0-1）
    :param glossiness: 光泽度（值越大，高光越集中）
    :param env_color: 环境反光色（BGR格式，模拟环境色反射）
    """
    try:
        # 1. 读取图片并转为浮点型（方便计算）
        image = cv2.imread(image_path).astype(np.float32) / 255.0
        if image is None:
            raise Exception("无法读取图片")
        h, w = image.shape[:2]

        # 2. 计算表面法线（模拟青蛙皮肤的曲面，用高斯模糊+梯度计算）
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # 计算x/y方向梯度（代表表面法线的x/y分量）
        grad_x = cv2.Sobel(blur_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur_gray, cv2.CV_32F, 0, 1, ksize=3)
        # 构造法线向量（归一化）
        norm = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1)
        normal = np.stack([-grad_x / norm, -grad_y / norm, 1 / norm], axis=-1)

        # 3. 归一化光源方向
        light_dir = np.array(light_dir, dtype=np.float32)
        light_dir = light_dir / np.linalg.norm(light_dir)

        # 4. 计算高光（Blinn-Phong模型，适合偏振反光）
        # 半程向量（视线方向默认朝画面内，即(0,0,1)）
        view_dir = np.array([0, 0, 1], dtype=np.float32)
        half_dir = (light_dir + view_dir) / np.linalg.norm(light_dir + view_dir)
        # 计算高光强度（贴合曲面）
        specular = np.power(np.clip(np.sum(normal * half_dir, axis=-1), 0, 1), glossiness)
        # 叠加环境反光色
        highlight = (np.ones((h, w, 3)) * np.array(env_color) / 255.0) + specular[..., None] * specular_strength
        highlight = np.clip(highlight, 0, 1)  # 限制范围

        # 5. 融合高光与原图（保留细节）
        result = image * (1 - specular_strength * specular[..., None]) + highlight * specular_strength * specular[
            ..., None]
        result = (np.clip(result, 0, 1) * 255).astype(np.uint8)

        # 6. 创建文件夹并保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"真实偏振高光已保存至：{output_path}")
        return result

    except Exception as e:
        print(f"错误：{str(e)}")
        return None


# ---------------------- 调用（针对这张青蛙图） ----------------------
if __name__ == "__main__":
    add_realistic_polar_highlight(
        image_path="/root/Polar3D-2/examples/images/004.png",  # 替换为你的青蛙图片路径
        output_path="/root/Polar3D-2/examples/HLT_images/004.png",
        light_dir=(0.5, -0.3, 1.0),
        specular_strength=0.7,
        glossiness=15,
        env_color=(10, 20, 30)
    )