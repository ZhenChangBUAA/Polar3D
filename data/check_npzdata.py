import numpy as np
import os

# ================= 配置区 =================
# 请替换为实际的文件路径
EXAMPLE_PATH = "/root/Polar3D-2/data/shape_diffusion/objaverse/surfaces/000-000/00a1a602456f4eb188b522d7ef19e81b.npz"  # 官方示例
MY_DATA_PATH = "/root/dataset/Objaverse-3D/003199cc6ff2410cb2d8e6f8a9cbb163/samples.npz"  # 你的数据

def analyze_npz(path, label):
    print(f"\n{'=' * 20} 分析数据: {label} {'=' * 20}")
    print(f"📂 文件路径: {path}")

    if not os.path.exists(path):
        print("❌ 文件不存在！请检查路径。")
        return None

    try:
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())
        print(f"🔑 包含 Keys: {keys}")

        stats = {}

        # 遍历检查每个 Key
        for k in keys:
            val = data[k]

            # 如果是标量或非数组
            if not isinstance(val, (np.ndarray, np.generic)):
                print(f"   - [{k}]: 类型 {type(val)} (非数组)")
                continue

            # 获取基本信息
            shape = val.shape
            dtype = val.dtype

            info_str = f"   - [{k}]: Shape={shape}, Dtype={dtype}"

            # 如果是数值型数组，计算统计量
            if np.issubdtype(dtype, np.number):
                v_min = np.min(val)
                v_max = np.max(val)
                v_mean = np.mean(val)
                v_std = np.std(val)

                info_str += f" | Min={v_min:.4f}, Max={v_max:.4f}, Mean={v_mean:.4f}"

                # 记录关键统计量用于对比
                stats[k] = {
                    "shape": shape,
                    "min": v_min,
                    "max": v_max,
                    "mean": v_mean
                }

                # 🚨 重点检查坐标范围 (假设 point_cloud 或 sdf 包含空间坐标)
                if k in ['point_cloud', 'points', 'coords', 'sdf', 'mesh_pos']:
                    range_val = v_max - v_min
                    info_str += f" | ⚠️ Range={range_val:.4f}"

                    if range_val > 1.1:
                        info_str += " [警告: 范围 > 1.0, 需确认是否归一化]"

            print(info_str)

        return stats

    except Exception as e:
        print(f"❌ 读取出错: {e}")
        return None


def compare_datasets(stats_ref, stats_my):
    print(f"\n{'=' * 20} ⚔️ 对比分析结果 ⚔️ {'=' * 20}")

    if stats_ref is None or stats_my is None:
        print("无法对比，因为有文件读取失败。")
        return

    # 1. 检查 Key 是否缺失
    keys_ref = set(stats_ref.keys())
    keys_my = set(stats_my.keys())

    missing_keys = keys_ref - keys_my
    extra_keys = keys_my - keys_ref

    if missing_keys:
        print(f"❌ 你的数据缺少关键 Key: {missing_keys}")
    if extra_keys:
        print(f"ℹ️ 你的数据包含额外 Key: {extra_keys}")

    # 2. 对比数值范围 (这是导致 NaN 的主要原因)
    common_keys = keys_ref.intersection(keys_my)
    print("\n🔍 数值范围对比 (Scale Check):")

    for k in common_keys:
        ref = stats_ref[k]
        my = stats_my[k]

        # 对比 Min/Max
        # 如果你的数据范围比示例大 10 倍以上，就是问题所在
        scale_ratio = (my['max'] - my['min']) / (ref['max'] - ref['min'] + 1e-6)

        status = "✅ 正常"
        if scale_ratio > 1.5:
            status = f"❌ 偏大 ({scale_ratio:.1f}x)"
        elif scale_ratio < 0.5:
            status = f"⚠️ 偏小 ({scale_ratio:.1f}x)"

        print(
            f"   [{k}]: 官方范围 [{ref['min']:.2f}, {ref['max']:.2f}] vs 你的范围 [{my['min']:.2f}, {my['max']:.2f}] -> {status}")


if __name__ == "__main__":
    stats_example = analyze_npz(EXAMPLE_PATH, "官方示例数据")
    stats_my = analyze_npz(MY_DATA_PATH, "我的数据")

    compare_datasets(stats_example, stats_my)