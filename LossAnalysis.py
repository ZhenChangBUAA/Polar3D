import pandas as pd
import matplotlib.pyplot as plt
import os

# ************************** 配置项 **************************
CSV_PATH = "/root/Polar3D-2/Train/3viewsRandom/outputs/step1x-3d-geometry/dinov2reglarge518-fluxflow-dit1300m/michelangelo-autoencoder+n32768+AdamWlr0.0001/csv_logs/version_4/metrics.csv"  # 完整CSV路径
SAVE_DIR = "/root/Polar3D-2/LossAnalysis/"  # 图片保存目录
TARGET_LOSS_COL = "train/loss_diffusion"  # 要分析的核心loss列
STEP_COL = "step"  # step列名
GROUP_AVG_SIZE = 10  # 功能2中每组数据的平均窗口大小（50个数据点）
# ************************************************************

# 1. 自动创建保存文件夹（如果不存在）
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Folder is ready: {SAVE_DIR}")

# 2. 读取CSV文件，并仅保留有效数据行（排除列名后的偶数行）
try:
    # 读取时保留原始行索引（方便后续筛选有效行）
    df = pd.read_csv(CSV_PATH, index_col=None)
    print(f"Successfully read CSV file, total {len(df)} rows of raw data")

    # 验证必要列是否存在
    required_cols = [STEP_COL, TARGET_LOSS_COL]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV file missing required column: {col}")

    # 关键筛选：排除列名后，仅保留偶数行（df数据行索引1、3、5...）
    valid_row_indices = [i for i in df.index if i % 2 == 1]
    df_valid = df.loc[valid_row_indices].reset_index(drop=True)

    # 过滤掉step或loss为空的数据（避免None值）
    df_valid = df_valid.dropna(subset=[STEP_COL, TARGET_LOSS_COL])
    # 确保step为数值类型（避免分组出错）
    df_valid[STEP_COL] = pd.to_numeric(df_valid[STEP_COL], errors="coerce")
    df_valid = df_valid.dropna(subset=[STEP_COL])

    print(f"After filtering, {len(df_valid)} rows of valid data (only even rows after header, no empty values)")
    if len(df_valid) == 0:
        raise ValueError("No valid data after filtering, please check CSV file format")

except FileNotFoundError:
    print(f"Error: File not found {CSV_PATH}")
    exit(1)
except Exception as e:
    print(f"Error: Failed to read/filter CSV file - {e}")
    exit(1)

# 3. 功能1：每100个step计算平均值并绘制折线图
print("\nStart processing Function 1: Calculate average loss every 100 steps")
# 3.1 按每100个step分组，计算平均值
df_valid["step_group_100"] = (df_valid[STEP_COL] // 100) * 100
loss_100_avg = df_valid.groupby("step_group_100").agg(
    avg_loss=(TARGET_LOSS_COL, "mean")
).reset_index()

# 3.2 绘制折线图并保存（全英文，无中文）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]  # 英文无字体问题
plt.figure(figsize=(12, 6))
plt.plot(loss_100_avg["step_group_100"], loss_100_avg["avg_loss"], color="#2F4F4F", linewidth=1.5)
plt.xlabel("Step (Grouped by 100 Steps)", fontsize=12)
plt.ylabel("Average Train Loss Diffusion", fontsize=12)
plt.title("Loss Curve (Average per 100 Steps)", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()

# 保存图片
fig1_path = os.path.join(SAVE_DIR, "loss_100step_average.png")
plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Function 1 image saved: {fig1_path}")

# 4. 功能2：按间隔6抽取loss，每组数据每50个点取均值，绘制6张子图（全英文）
print(
    "\nStart processing Function 2: Extract loss by interval 6, calculate average per 50 points for each group, plot 6 subplots")
# 4.1 按间隔6抽取数据，生成6组原始数据
df_valid["mod_6"] = df_valid.index % 6
loss_raw_groups = []
for i in range(6):
    group_data = df_valid[df_valid["mod_6"] == i].reset_index(drop=True)
    loss_raw_groups.append(group_data)
    print(f"  Raw Group {i} has {len(group_data)} rows of data")

# 4.2 对每组原始数据，按每50个数据点计算平均值（平滑处理）
loss_smoothed_groups = []
for i, group_data in enumerate(loss_raw_groups):
    if len(group_data) == 0:
        # 空分组直接存入空DataFrame
        loss_smoothed_groups.append(pd.DataFrame(columns=["inner_group_50", "avg_loss", "representative_step"]))
        print(f"  Group {i} has no valid data, skip average calculation")
        continue

    # 为每组数据创建内部分组标识（每50个数据点为一组）
    group_data["inner_group_50"] = (group_data.index // GROUP_AVG_SIZE) * GROUP_AVG_SIZE

    # 分组计算平均值，同时保留代表性step（取每组第一个step，用于绘图x轴）
    smoothed_group = group_data.groupby("inner_group_50").agg(
        avg_loss=(TARGET_LOSS_COL, "mean"),
        representative_step=(STEP_COL, "first")  # 英文列名，避免后续报错
    ).reset_index()

    loss_smoothed_groups.append(smoothed_group)
    print(
        f"  Group {i} has {len(smoothed_group)} average data points after smoothing (average per {GROUP_AVG_SIZE} points)")

# 4.3 绘制6张子图（2行3列）并保存（全英文）
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
colors = ["#FF6347", "#4169E1", "#32CD32", "#FFD700", "#9370DB", "#FF69B4"]

for i, (smoothed_group, ax, color) in enumerate(zip(loss_smoothed_groups, axes, colors)):
    if len(smoothed_group) == 0:
        ax.text(0.5, 0.5, "No Valid Data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Loss Group {i} (No Valid Data)", fontsize=12, fontweight="bold")
    else:
        # 绘制平滑后的折线图（引用正确的英文列名representative_step，解决KeyError）
        ax.plot(smoothed_group["representative_step"], smoothed_group["avg_loss"], color=color, linewidth=1.5,
                alpha=0.8)
        # 设置子图标题和标签（全英文）
        ax.set_title(f"Loss Group {i} (Avg per {GROUP_AVG_SIZE} Points)", fontsize=12, fontweight="bold")

    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Train Loss Diffusion (Avg)", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

# 调整子图间距
plt.tight_layout()
plt.subplots_adjust(top=0.92)
fig.suptitle(f"Loss Curves by Interval 6 (6 Groups, Avg per {GROUP_AVG_SIZE} Points)", fontsize=16, fontweight="bold")

# 保存图片
fig2_path = os.path.join(SAVE_DIR, f"loss_interval_6_groups_{GROUP_AVG_SIZE}avg.png")
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Function 2 image saved: {fig2_path}")

print("\nAll tasks completed! All images are saved in: ", SAVE_DIR)