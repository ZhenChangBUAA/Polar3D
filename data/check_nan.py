import numpy as np
import os
import glob
from tqdm import tqdm
import traceback

# ================= 配置区 =================
# 设置你要检查的数据集根目录
DATA_ROOT = "/root/dataset/Objaverse-3D/npz"


# =========================================

def check_nan_in_file(file_path):
    """
    检查单个文件，返回 (是否损坏, 是否含NaN, 含NaN的Keys列表, 错误信息)
    """
    try:
        # 使用上下文管理器加载，确保文件句柄被释放
        with np.load(file_path, allow_pickle=True) as data:
            keys_with_nan = []
            keys_with_inf = []

            for k in data.files:
                try:
                    arr = data[k]

                    # 1. 检查是否为数值类型 (跳过文件名、字符串等 metadata)
                    if not np.issubdtype(arr.dtype, np.number):
                        continue

                    # 2. 检查 NaN
                    if np.isnan(arr).any():
                        keys_with_nan.append(k)

                    # 3. 检查 Inf (无穷大同样会导致训练崩坏)
                    if np.isinf(arr).any():
                        keys_with_inf.append(k)

                except Exception:
                    # 极少数情况读取某个key出错
                    continue

            if keys_with_nan or keys_with_inf:
                return False, True, keys_with_nan + keys_with_inf, None
            else:
                return False, False, [], None

    except Exception as e:
        # 文件彻底损坏，无法 load
        return True, False, [], str(e)


def batch_check_nan():
    print(f"🚀 开始扫描目录: {DATA_ROOT}")

    # 1. 查找所有 npz 文件
    # recursive=True 确保子文件夹也被扫描
    search_pattern = os.path.join(DATA_ROOT, "**/*.npz")
    files = glob.glob(search_pattern, recursive=True)

    if not files:
        print("❌ 未找到任何 .npz 文件，请检查路径。")
        return

    print(f"🔍 共找到 {len(files)} 个文件，开始逐个检查...")

    bad_files_nan = []  # 包含 NaN 的文件
    bad_files_corrupt = []  # 无法读取的文件

    # 使用 tqdm 显示进度条
    for file_path in tqdm(files, desc="Checking"):
        is_corrupt, has_nan, bad_keys, err_msg = check_nan_in_file(file_path)

        if is_corrupt:
            bad_files_corrupt.append((file_path, err_msg))
        elif has_nan:
            bad_files_nan.append((file_path, bad_keys))

    # ================= 📊 最终报告 =================
    print("\n" + "=" * 40)
    print("📋 检查结果汇总报告")
    print("=" * 40)
    print(f"✅ 总扫描文件: {len(files)}")
    print(f"✅ 正常文件:   {len(files) - len(bad_files_nan) - len(bad_files_corrupt)}")
    print(f"❌ 含 NaN/Inf: {len(bad_files_nan)}")
    print(f"💀 文件损坏:   {len(bad_files_corrupt)}")
    print("=" * 40)

    if bad_files_nan:
        print("\n⚠️  以下文件包含 NaN 或 Inf (建议删除或重新生成):")
        for f, keys in bad_files_nan:
            print(f"  📄 {f}")
            print(f"     └─ 问题 Keys: {keys}")

    if bad_files_corrupt:
        print("\n💀 以下文件已损坏 (无法读取):")
        for f, err in bad_files_corrupt:
            print(f"  📄 {f}")
            print(f"     └─ 错误: {err}")

    # 生成一个清理脚本建议
    if bad_files_nan or bad_files_corrupt:
        print("\n💡 提示: 是否要生成一个删除这些坏文件的脚本？(y/n)")
        # 这一步需要你手动决定，防止脚本误删
        # 你可以根据输出手动处理


if __name__ == "__main__":
    batch_check_nan()