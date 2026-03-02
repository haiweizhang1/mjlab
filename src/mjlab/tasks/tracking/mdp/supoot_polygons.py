import numpy as np
from scipy.spatial import ConvexHull, QhullError
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ==========================================
# 1. 核心计算模块
# ==========================================

def compute_support_polygons_batch(contact_points_batch, contact_mask_batch):
    """
    批量计算双足机器人的支撑域凸包。

    Args:
        contact_points_batch (np.ndarray): 形状为 (Batch_Size, Max_Points, 2) 的坐标数据。
                                           例如 (4096, 12, 2)。
        contact_mask_batch (np.ndarray): 形状为 (Batch_Size, Max_Points) 的布尔或二值掩码。
                                         1 (True) 表示该点当前真实接触地面。

    Returns:
        list: 一个长度为 Batch_Size 的列表。列表中的每个元素都是一个 numpy 数组，
              表示该帧支撑域顶点的有序坐标 (按逆时针排列)，形状为 (N_vertices, 2)。
              注意：如果接触点少于3个，则返回这些点本身。
    """
    batch_size = contact_points_batch.shape[0]
    support_polygons_list = []

    for i in range(batch_size):
        # --- 步骤 1: 数据清洗 ---
        points_this_frame = contact_points_batch[i]  # shape (12, 2)
        mask_this_frame = contact_mask_batch[i]  # shape (12,)

        # 提取有效接触点
        # 假设 mask > 0.5 为有效接触
        valid_points = points_this_frame[mask_this_frame > 0.5]

        # [重要] 去重。如果多个传感器映射到同一个物理位置，不去重可能会导致 Qhull 错误。
        valid_points = np.unique(valid_points, axis=0)

        num_valid = valid_points.shape[0]

        # --- 步骤 2: 处理退化情况 (点数少于3) ---
        if num_valid < 3:
            # 0个点(空中)，1个点(点接触)，2个点(线接触)无法构成多边形区域。
            # 直接返回这些点作为支撑元素的表达。
            support_polygons_list.append(valid_points)
            continue

        # --- 步骤 3: 计算凸包 ---
        try:
            # 计算凸包
            hull = ConvexHull(valid_points)

            # hull.vertices 提供了构成凸包的点的索引，并且保证是逆时针顺序
            hull_vertices_coords = valid_points[hull.vertices]

            support_polygons_list.append(hull_vertices_coords)

        except QhullError:
            # [重要] 处理异常：例如所有有效点完美共线（这在仿真中很常见）。
            # 在这种情况下，凸包退化为一条线段。我们降级为返回所有有效点。
            print(f"Warning: Frame {i} points are collinear. Returning all points.")
            support_polygons_list.append(valid_points)

    return support_polygons_list


# ==========================================
# 2. 可视化验证模块
# ==========================================

def visualize_support_frame(all_points, mask, hull_polygon, frame_idx):
    """
    可视化单帧的接触状态和计算出的支撑域。
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. 绘制所有潜在的传感器点位 (灰色空心)
    ax.scatter(all_points[:, 0], all_points[:, 1],
               s=100, facecolors='none', edgecolors='gray', linestyle='--', label='Potential Points')

    # 2. 绘制实际接触点 (红色实心)
    valid_points = all_points[mask > 0.5]
    if len(valid_points) > 0:
        ax.scatter(valid_points[:, 0], valid_points[:, 1],
                   s=100, c='red', marker='o', label='Active Contacts')

    # 3. 绘制支撑域凸包 (蓝色区域)
    N_hull = len(hull_polygon)
    title_suffix = ""
    if N_hull >= 3:
        # 创建多边形补片
        poly_patch = patches.Polygon(hull_polygon, closed=True,
                                     edgecolor='blue', facecolor='cyan',
                                     alpha=0.3, linewidth=2, label='Support Polygon (Convex Hull)')
        ax.add_patch(poly_patch)
        title_suffix = "(Stable Polygon Support)"
    elif N_hull == 2:
        # 线接触
        ax.plot(hull_polygon[:, 0], hull_polygon[:, 1], 'b-', linewidth=3, label='Line Support')
        title_suffix = "(Unstable Line Support)"
    elif N_hull == 1:
        title_suffix = "(Unstable Point Support)"
    else:
        title_suffix = "(No Contact - Flying)"

    # 设置图表属性
    ax.set_title(f"Frame {frame_idx} Analysis\n{title_suffix}")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.legend()
    ax.grid(True, linestyle=':')
    ax.set_aspect('equal', 'box')  # 保证坐标轴比例一致，否则形状会变形

    # 设置坐标轴范围以便观察 (根据模拟数据调整)
    ax.set_xlim(-0.3, 0.7)
    ax.set_ylim(-0.4, 0.4)

    plt.show()


# ==========================================
# 3. 数据模拟与主程序测试
# ==========================================

def generate_mock_biped_data(batch_size=4096):
    """
    生成模拟双足机器人的数据用于测试。
    包括两只脚的固定位置和随机的接触状态。
    """
    # 定义左脚和右脚相对于身体中心的标准位置 (假设每只脚6个点：4角+2中间)
    # 左脚中心大概在 y=+0.15, 右脚在 y=-0.15
    left_foot_template = np.array([
        [0.1, 0.1], [0.1, 0.2], [-0.1, 0.1], [-0.1, 0.2],  # 角点
        [0.0, 0.1], [0.0, 0.2]  # 中间点
    ])
    right_foot_template = np.array([
        [0.1, -0.2], [0.1, -0.1], [-0.1, -0.2], [-0.1, -0.1],
        [0.0, -0.2], [0.0, -0.1]
    ])

    # 组合成 12 个点的模板
    points_template = np.vstack([left_foot_template, right_foot_template])  # (12, 2)

    # 复制到整个 batch
    points_batch = np.tile(points_template, (batch_size, 1, 1))

    # 加上一些随机噪声模拟测量误差
    points_batch += np.random.normal(0, 0.005, points_batch.shape)

    # 生成随机 Mask 来模拟不同的行走阶段
    mask_batch = np.zeros((batch_size, 12))

    for i in range(batch_size):
        phase = np.random.rand()
        if phase < 0.4:
            # 左脚单支撑 (左侧6个点随机接触4-6个)
            mask_batch[i, 0:6] = np.random.rand(6) > 0.2
        elif phase < 0.8:
            # 右脚单支撑
            mask_batch[i, 6:12] = np.random.rand(6) > 0.2
        elif phase < 0.95:
            # 双脚支撑 (所有点随机接触)
            mask_batch[i, :] = np.random.rand(12) > 0.3
        else:
            # 跳跃/腾空 (几乎无接触)
            mask_batch[i, :] = np.random.rand(12) > 0.9

    return points_batch, mask_batch


if __name__ == "__main__":
    # 1. 生成模拟输入数据
    print("正在生成模拟数据 (4096帧)...")
    input_points, input_mask = generate_mock_biped_data(4096)
    print(f"输入数据形状: Points={input_points.shape}, Mask={input_mask.shape}")

    # 2. 执行计算
    print("\n正在计算所有帧的凸包...")
    results = compute_support_polygons_batch(input_points, input_mask)
    print("计算完成。")

    # 3. 挑选几个典型帧进行可视化验证
    print("\n开始可视化验证 (关闭弹窗以查看下一帧)...")

    # 挑选不同类型的帧进行展示
    frames_to_visualize = []

    # 找一个双脚支撑的帧 (顶点数通常 > 6)
    for i, res in enumerate(results):
        if len(res) >= 7:
            frames_to_visualize.append(i)
            break

    # 找一个单脚支撑的帧 (顶点数通常为 4-5)
    for i, res in enumerate(results):
        if 4 <= len(res) <= 5:
            frames_to_visualize.append(i)
            break

    # 找一个退化情况的帧 (例如只有2点线接触)
    for i, res in enumerate(results):
        if len(res) == 2:
            frames_to_visualize.append(i)
            break

    # 如果没找到特殊的，就随机看前几个
    if not frames_to_visualize:
        frames_to_visualize = [0, 100, 500]

    for frame_idx in frames_to_visualize:
        print(f"可视化第 {frame_idx} 帧: 计算出的凸包顶点数 = {len(results[frame_idx])}")
        visualize_support_frame(
            input_points[frame_idx],
            input_mask[frame_idx],
            results[frame_idx],
            frame_idx
        )

    print("\n验证结束。")