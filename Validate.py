import warnings
import argparse
import os

import numpy as np
import torch
import json
import sys

import trimesh

# 忽略无关警告
warnings.filterwarnings("ignore")

# ========== 核心新增：导入 DeepSpeed Zero 权重转换工具 ==========
try:
    # 确保 zero_to_fp32.py 在 Python 路径中（若不在需添加路径）
    # 若提示找不到该文件，执行：sys.path.append("存放zero_to_fp32.py的目录")
    from zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
except ImportError as e:
    print(f"【警告】未找到 zero_to_fp32.py，尝试从 DeepSpeed 源码加载...")
    # 备选：从 DeepSpeed 官方路径加载（若已安装 deepspeed）
    try:
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    except ImportError:
        raise ImportError(
            "无法导入 get_fp32_state_dict_from_zero_checkpoint！\n"
            "请确保 zero_to_fp32.py 在 Python 路径中，或安装 deepspeed：pip install deepspeed"
        )

# 导入项目核心模块
from omegaconf import OmegaConf
import step1x3d_geometry
from step1x3d_geometry.systems.base import BaseSystem
from step1x3d_geometry.utils.config import load_config


def load_deepspeed_zero_ckpt(ckpt_dir: str) -> dict:
    """
    使用 DeepSpeed 官方工具解析 Zero 分布式 ckpt 文件夹，提取 fp32 格式的 state_dict
    :param ckpt_dir: 分布式 ckpt 文件夹路径（如 epoch=2-step=45.ckpt）
    :return: 纯 fp32 格式的模型权重字典（无 module. 前缀，可直接加载）
    """
    # 分离 ckpt 根目录和标签（适配 zero_to_fp32 工具的参数要求）
    # 示例：ckpt_dir = /xxx/epoch=2-step=45.ckpt → output_dir=/xxx, tag=epoch=2-step=45.ckpt
    output_dir, tag = os.path.split(ckpt_dir)
    if not tag:  # 若路径以 / 结尾，重新拆分
        output_dir, tag = os.path.split(output_dir)

    print(f"===== 解析 DeepSpeed Zero 分布式 ckpt =====")
    print(f"ckpt 根目录：{output_dir}")
    print(f"ckpt 标签：{tag}")

    # 核心：用官方工具提取 fp32 权重（自动处理 Zero 分片、移除 module. 前缀）
    state_dict = get_fp32_state_dict_from_zero_checkpoint(output_dir, tag=tag)
    print(f"✅ 成功提取 fp32 权重，共 {len(state_dict.keys())} 个参数")

    return state_dict


def load_model_from_deepspeed_ckpt(
        system_class,
        ckpt_path: str,
        cfg,
        device: torch.device
) -> BaseSystem:
    """
    适配：普通单文件ckpt / DeepSpeed Zero 分布式文件夹 ckpt
    """
    # 步骤1：实例化模型（用配置初始化）
    system = system_class(cfg=cfg, resumed=True)
    system.to(device)
    system.eval()  # 推理模式

    # 步骤2：判断 ckpt 类型并加载权重
    if os.path.isdir(ckpt_path):
        print(f"检测到 ckpt 是 DeepSpeed Zero 分布式文件夹：{ckpt_path}")
        # 用官方工具提取 fp32 权重
        state_dict = load_deepspeed_zero_ckpt(ckpt_path)
        # 加载权重到模型（strict=False 忽略训练相关的不匹配键）
        msg = system.load_state_dict(state_dict, strict=False)
    else:
        print(f"检测到 ckpt 是普通单文件：{ckpt_path}")
        # 普通 ckpt 直接加载
        system = system_class.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=cfg,
            map_location=device,
            resumed=True
        )
        msg = None  # load_from_checkpoint 自动处理，无需手动返回

    # 打印权重加载信息
    if msg is not None:
        print(f"===== 权重加载详情 =====")
        print(f"  - 未加载的键（可忽略）：{msg.missing_keys[:5]}（仅显示前5个）")
        print(f"  - 不匹配的键（可忽略）：{msg.unexpected_keys[:5]}（仅显示前5个）")
    print(f"模型加载完成（ckpt 路径：{ckpt_path}）")

    return system


def check_path_valid(path: str, desc: str, is_file: bool = True):
    """路径合法性检查"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"【错误】{desc}路径不存在：{path}")
    if is_file and os.path.isdir(path) and "ckpt" not in path.lower():
        raise IsADirectoryError(f"【错误】{desc}路径是普通目录（非ckpt文件夹）：{path}")


def geometry_pipeline_custom(
        input_image_path: str,
        save_glb_path: str,
        ckpt_path: str,
        config_path: str,
        gpu_ids: str = "0"
):
    """
    最终版：适配 DeepSpeed Zero 分布式 ckpt 的 3D 几何生成推理脚本
    """
    # -------------------------- 1. 基础检查 --------------------------
    check_path_valid(input_image_path, "输入图片")
    check_path_valid(config_path, "配置文件")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"【错误】CKPT路径不存在：{ckpt_path}")

    # -------------------------- 2. 环境配置 --------------------------
    # 强制单卡推理（DeepSpeed 权重转换后无需多卡）
    gpu_ids = "0" if gpu_ids else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"===== 环境配置 =====")
    print(f"使用设备: {device}")
    print(f"指定GPU: {gpu_ids}（DeepSpeed 权重适配后强制单卡）")

    # -------------------------- 3. 加载训练配置 --------------------------
    class MockArgs:
        config = config_path
        local_rank = 0
        verbose = False
        typecheck = False

    args = MockArgs()
    extras = []
    cfg = load_config(args.config, cli_args=extras, n_gpus=1)  # 强制单卡配置
    print(f"\n===== 配置加载 =====")
    print(f"配置文件加载完成：{config_path}")

    # -------------------------- 4. 加载 DeepSpeed Zero 模型（核心） --------------------------
    system_class = step1x3d_geometry.find(cfg.system_type)
    print(f"\n===== 模型加载 =====")
    system = load_model_from_deepspeed_ckpt(
        system_class=system_class,
        ckpt_path=ckpt_path,
        cfg=cfg.system,
        device=device
    )
    system.to(device)
    system.eval()  # 确保推理模式

    # -------------------------- 5. 构造推理输入 --------------------------
    print(f"\n===== 执行推理 =====")
    sample_inputs = {
        "image": [input_image_path],
        # 按需补充训练时用到的条件（如caption/label）
        # "caption": ["a chair"],
        # "label": [{"symmetry": "none", "edge_type": "soft"}],
    }
    # 固定随机种子（保证可复现）
    torch.manual_seed(2025)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2025)

    # 生成 3D Latent
    out = system.sample(
        sample_inputs=sample_inputs,
        sample_times=1,
        steps=cfg.system.num_inference_steps,
        guidance_scale=cfg.system.guidance_scale,
        eta=cfg.system.eta,
        seed=2025,
    )
    print("3D Latent 生成完成")

    # -------------------------- 6. 提取 Mesh 并保存 GLB --------------------------
    latents = out["latents"][0]
    mesh_extract_results  = system.shape_model.extract_geometry(
        latents,
        bounds=1.05,
        mc_level=0.0,
        octree_resolution=256,
        enable_pbar=False,
    )
    for idx, result in enumerate(mesh_extract_results):
        if result is None:
            print(f"⚠️  第{idx}个Mesh提取失败，跳过")
            continue
        verts = result.verts.cpu().detach().numpy()  # (N, 3) 顶点坐标
        faces = result.faces.cpu().detach().numpy()
        faces = faces.astype(np.int32)
        if faces.min() == 1:  # 若面索引从1开始，转为从0开始（trimesh要求）
            faces = faces - 1

        # 步骤3：创建trimesh Mesh对象（核心：这是可export的对象）
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # 步骤4：保存为GLB文件
        if idx == 0:  # 仅保存第一个Mesh，或遍历保存所有
            os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
            mesh.export(save_glb_path, file_type="glb")  # 指定格式为GLB
            print(f"✅ 第{idx}个Mesh已保存至：{save_glb_path}")
            break

    print(f"\n===== 推理完成 =====")
    print(f"GLB 文件已保存至：{save_glb_path}")


if __name__ == "__main__":
    # -------------------------- 命令行参数解析 --------------------------
    parser = argparse.ArgumentParser(description="3D几何生成推理脚本（适配 DeepSpeed Zero 权重）")
# a2d765d02050414d958cdf0209a8899e-2850.glb d930940ff5b449a4a2d167d6210d13a3-2850.glb 8a3f51d301974cab8b5761119ead0ead-2850.glb
  # 00cd8d7fba924ff4861936d32baef310
    #014e9cc64c414162893ab309f6adc944
    # 0079418517784f2bbf0aec6fd8de3cde

    # 35b8375cb3ac442aa36af8108efc6da8
    # 核心推理参数（必选）
    parser.add_argument("--input_image",
                        default="/root/dataset/Objaverse-2D/014e9cc64c414162893ab309f6adc944/normals_000_front.png",
                        help="输入测试图片的路径")
    parser.add_argument("--save_glb",
                        default="/root/Polar3D-2/Test/LoraNormal/014e9cc64c414162893ab309f6adc944-lora2.0.glb",
                        help="生成的GLB文件保存路径")
    parser.add_argument("--ckpt_path",
                        default="/root/Polar3D-2/Train/qiuyu/outputs/step1x-3d-geometry/dinov2reglarge518-fluxflow-dit1300m/michelangelo-autoencoder+n32768+AdamWlr0.0001/ckpts/epoch=583-step=3500.ckpt/checkpoint",
                        help="DeepSpeed Zero 分布式 ckpt 文件夹路径")
    parser.add_argument("--config_path",
                        default="/root/Polar3D-2/configs/train-geometry-diffusion/step1x-3d-geometry-1300m.yaml",
                        help="训练时使用的配置文件路径")

    # 可选参数（强制单卡，无需多卡）
    parser.add_argument("--gpu",
                        default="0",
                        help="指定使用的GPU ID（仅支持单卡，默认 0）")

    # 解析参数
    args = parser.parse_args()

    # 执行推理
    try:
        geometry_pipeline_custom(
            input_image_path=args.input_image,
            save_glb_path=args.save_glb,
            ckpt_path=args.ckpt_path,
            config_path=args.config_path,
            gpu_ids=args.gpu
        )
    except Exception as e:
        print(f"\n【执行失败】{type(e).__name__}: {str(e)[:500]}")  # 截断过长的错误信息
        # 打印完整堆栈（方便调试）
        import traceback
        traceback.print_exc()
        exit(1)