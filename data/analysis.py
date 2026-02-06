import trimesh
import numpy as np
import argparse
from pathlib import Path


def load_glb_mesh(file_path):
    """
    加载GLB文件并提取Mesh数据

    参数:
        file_path: GLB文件路径
    返回:
        trimesh.Trimesh对象
    """
    try:
        # 加载GLB文件
        scene = trimesh.load(file_path, file_type='glb')

        # 如果是场景对象，合并所有mesh
        if isinstance(scene, trimesh.Scene):
            # 合并场景中的所有几何体
            meshes = []
            for geom in scene.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom)

            if not meshes:
                raise ValueError(f"文件 {file_path} 中未找到Mesh数据")

            # 合并所有mesh
            combined_mesh = trimesh.util.concatenate(meshes)
            return combined_mesh
        elif isinstance(scene, trimesh.Trimesh):
            return scene
        else:
            raise ValueError(f"不支持的文件类型: {type(scene)}")

    except Exception as e:
        raise RuntimeError(f"加载文件 {file_path} 失败: {str(e)}")


def calculate_mesh_center(mesh):
    """
    计算Mesh的中心点（包围盒中心）

    参数:
        mesh: trimesh.Trimesh对象
    返回:
        中心点坐标 (x, y, z)
    """
    # 获取包围盒的最小值和最大值
    min_bound = mesh.bounds[0]
    max_bound = mesh.bounds[1]

    # 计算中心点
    center = (min_bound + max_bound) / 2
    return tuple(center)


def calculate_sphere_radius(all_vertices, sphere_center):
    """
    计算包围所有顶点的最小球体半径

    参数:
        all_vertices: 所有Mesh的顶点数组
        sphere_center: 球体中心点
    返回:
        球体半径
    """
    # 计算每个顶点到中心点的距离
    distances = np.linalg.norm(all_vertices - sphere_center, axis=1)

    # 最大距离即为半径（添加少量余量）
    radius = np.max(distances) * 1.05  # 5% 余量
    return radius


def analyze_glb_files(file_paths):
    """
    分析多个GLB文件的Mesh数据

    参数:
        file_paths: GLB文件路径列表
    """
    # 存储所有Mesh的信息
    mesh_infos = []
    all_vertices = []

    print("=" * 60)
    print("开始分析GLB文件...")
    print("=" * 60)

    # 处理每个文件
    for i, file_path in enumerate(file_paths):
        print(f"\n[{i}] 处理文件: {file_path}")

        # 验证文件存在
        if not Path(file_path).exists():
            print(f"错误：文件 {file_path} 不存在！")
            continue

        try:
            # 加载Mesh
            mesh = load_glb_mesh(file_path)

            # 计算中心点
            center = calculate_mesh_center(mesh)

            # 存储Mesh信息
            mesh_info = {
                'file': file_path,
                'center': center,
                'vertex_count': len(mesh.vertices),
                'bounds': mesh.bounds
            }
            mesh_infos.append(mesh_info)

            # 收集顶点数据
            all_vertices.extend(mesh.vertices)

            # 输出当前Mesh信息
            print(f"  顶点数量: {mesh_info['vertex_count']}")
            print(f"  包围盒范围:")
            print(
                f"    最小值: ({mesh_info['bounds'][0][0]:.4f}, {mesh_info['bounds'][0][1]:.4f}, {mesh_info['bounds'][0][2]:.4f})")
            print(
                f"    最大值: ({mesh_info['bounds'][1][0]:.4f}, {mesh_info['bounds'][1][1]:.4f}, {mesh_info['bounds'][1][2]:.4f})")
            print(f"  Mesh中心点: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")

        except Exception as e:
            print(f"  处理失败: {str(e)}")
            continue

    if not mesh_infos:
        print("\n错误：没有成功加载任何Mesh数据！")
        return

    # 转换顶点数据为numpy数组
    all_vertices = np.array(all_vertices)

    # 计算包含所有Mesh的球体中心点（所有顶点的几何中心）
    sphere_center = np.mean(all_vertices, axis=0)

    # 计算球体半径
    sphere_radius = calculate_sphere_radius(all_vertices, sphere_center)

    # 输出最终结果
    print("\n" + "=" * 60)
    print("分析结果汇总")
    print("=" * 60)

    print("\n【各Mesh中心点】")
    for i, info in enumerate(mesh_infos, 1):
        print(f"[{i}] {Path(info['file']).name}:")
        print(f"    X: {info['center'][0]:.6f}")
        print(f"    Y: {info['center'][1]:.6f}")
        print(f"    Z: {info['center'][2]:.6f}")

    print("\n【包围球体参数】")
    print(f"球体中心点:")
    print(f"  X: {sphere_center[0]:.6f}")
    print(f"  Y: {sphere_center[1]:.6f}")
    print(f"  Z: {sphere_center[2]:.6f}")
    print(f"球体半径: {sphere_radius:.6f}")

    # 验证：计算每个Mesh中心点到球体中心的距离
    print("\n【验证】各Mesh中心点到球体中心的距离:")
    for i, info in enumerate(mesh_infos, 1):
        distance = np.linalg.norm(np.array(info['center']) - sphere_center)
        print(f"[{i}] {Path(info['file']).name}: {distance:.6f} (小于半径 {sphere_radius:.6f})")


def main():
    """主函数"""
    # 设置命令行参数
    # parser = argparse.ArgumentParser(description='分析GLB文件的Mesh中心点和包围球体半径')
    # parser.add_argument('files', help='输入3个GLB文件的路径', default=["/root/Polar3D/output/dragon.glb","/root/Polar3D/output/STBR.glb","/root/Polar3D/output/xiaofangshuan.glb"])
    # args = parser.parse_args()
    files = ["/root/dataset/Objaverse-3D/003199cc6ff2410cb2d8e6f8a9cbb163/watertight_mesh.obj"]
    # 执行分析
    analyze_glb_files(files)

if __name__ == "__main__":
    # 安装依赖提示
    print("提示：如果运行出错，请先安装依赖：")
    print("      pip install trimesh numpy\n")

    try:
        main()
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        print("\n使用方法:")
        print("  python glb_analyzer.py file1.glb file2.glb file3.glb")