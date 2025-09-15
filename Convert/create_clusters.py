import numpy as np
import json
from pathlib import Path
from scipy.spatial import cKDTree
from collections import defaultdict
import sys

# Lambda3 GPUパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_coordination_based_clusters(positions, cutoff=3.5, ideal_coord=8):
    """
    配位数ベースのクラスター生成
    BCC構造（配位数8）からの逸脱を検出
    """
    print("🔬 Creating coordination-based clusters...")

    n_atoms = positions.shape[0]
    tree = cKDTree(positions)

    # 各原子の近傍を探索
    neighbors = tree.query_ball_tree(tree, cutoff)
    coord_numbers = np.array([len(n) - 1 for n in neighbors])

    # 配位数による分類
    clusters = defaultdict(list)

    # 理想的なBCC構造（配位数8）
    perfect_bcc = np.where(coord_numbers == ideal_coord)[0]
    if len(perfect_bcc) > 0:
        clusters[0] = perfect_bcc.tolist()

    # 配位数欠陥（転位コア、空孔など）
    defect_types = {
        1: (0, 4),      # 深刻な欠陥
        2: (5, 6),      # 空孔周辺
        3: (7, 7),      # 軽度欠陥
        4: (9, 10),     # 格子間原子周辺
        5: (11, 20)     # 高密度領域
    }

    for cluster_id, (min_coord, max_coord) in defect_types.items():
        atoms = np.where((coord_numbers >= min_coord) &
                        (coord_numbers <= max_coord))[0]
        if len(atoms) > 0:
            clusters[cluster_id] = atoms.tolist()

    print(f"  Created {len(clusters)} coordination clusters")
    for cid, atoms in clusters.items():
        if cid == 0:
            print(f"    Cluster {cid}: {len(atoms)} atoms (perfect BCC)")
        else:
            print(f"    Cluster {cid}: {len(atoms)} atoms (defect type)")

    return dict(clusters)

def create_atomic_species_clusters(positions, atom_types):
    """
    原子種ベースのクラスター生成
    Fe主相、C炭化物、Cr炭化物などを分離
    """
    print("🧪 Creating atomic species clusters...")

    clusters = {}
    unique_types = np.unique(atom_types)

    for i, atype in enumerate(unique_types):
        atoms = np.where(atom_types == atype)[0]
        clusters[i] = atoms.tolist()
        print(f"  Cluster {i} ({atype}): {len(atoms)} atoms")

    return clusters

def create_spatial_grid_clusters(positions, n_per_dim=5):
    """
    空間グリッドベースのクラスター生成
    局所的な歪み・損傷解析用
    """
    print("📐 Creating spatial grid clusters...")

    # バウンディングボックス
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    box_size = max_pos - min_pos

    # グリッドサイズ
    grid_size = box_size / n_per_dim

    clusters = defaultdict(list)
    n_atoms = positions.shape[0]

    for i in range(n_atoms):
        # グリッドインデックス計算
        grid_idx = np.floor((positions[i] - min_pos) / grid_size).astype(int)
        grid_idx = np.clip(grid_idx, 0, n_per_dim - 1)

        # クラスターID計算（3D to 1D）
        cluster_id = (grid_idx[0] * n_per_dim**2 +
                     grid_idx[1] * n_per_dim +
                     grid_idx[2])

        clusters[cluster_id].append(i)

    # 空のクラスターを除去
    clusters = {k: v for k, v in clusters.items() if len(v) > 0}

    print(f"  Created {len(clusters)} spatial clusters")
    print(f"  Average atoms per cluster: {n_atoms / len(clusters):.1f}")

    return dict(clusters)

def create_defect_region_clusters(positions, cutoff=3.5, expansion_radius=5.0):
    """
    欠陥領域クラスター生成
    転位コアや亀裂先端周辺を重点的にクラスター化
    """
    print("💎 Creating defect region clusters...")

    n_atoms = positions.shape[0]
    tree = cKDTree(positions)

    # 配位数計算
    neighbors = tree.query_ball_tree(tree, cutoff)
    coord_numbers = np.array([len(n) - 1 for n in neighbors])

    # 欠陥原子の特定（配位数が8でない）
    defect_atoms = np.where(np.abs(coord_numbers - 8) > 1)[0]

    clusters = {}
    cluster_id = 0
    assigned = np.zeros(n_atoms, dtype=bool)

    # 各欠陥原子周辺をクラスター化
    for defect in defect_atoms:
        if assigned[defect]:
            continue

        # 欠陥周辺の原子を取得
        region_atoms = tree.query_ball_point(positions[defect], expansion_radius)

        # クラスターに追加
        clusters[cluster_id] = region_atoms
        assigned[region_atoms] = True
        cluster_id += 1

    # 残りの原子（健全領域）を大きなクラスターに
    healthy_atoms = np.where(~assigned)[0]
    if len(healthy_atoms) > 0:
        # 空間分割で健全領域をクラスター化
        healthy_positions = positions[healthy_atoms]
        n_healthy_clusters = max(5, len(healthy_atoms) // 500)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_healthy_clusters, random_state=42)
        labels = kmeans.fit_predict(healthy_positions)

        for i in range(n_healthy_clusters):
            cluster_atoms = healthy_atoms[labels == i]
            clusters[cluster_id] = cluster_atoms.tolist()
            cluster_id += 1

    print(f"  Created {len(clusters)} defect-focused clusters")
    print(f"  Defect region clusters: {len(defect_atoms)}")
    print(f"  Healthy region clusters: {cluster_id - len(defect_atoms)}")

    return clusters

def create_hybrid_clusters(positions, atom_types, method='balanced'):
    """
    ハイブリッドクラスター生成（推奨）
    複数の手法を組み合わせて材料科学的に意味のあるクラスターを生成
    """
    print("🔮 Creating hybrid clusters...")

    n_atoms = positions.shape[0]

    if method == 'balanced':
        # バランス型：配位数と空間を組み合わせ

        # Step 1: 配位数で大分類
        tree = cKDTree(positions)
        neighbors = tree.query_ball_tree(tree, 3.5)
        coord_numbers = np.array([len(n) - 1 for n in neighbors])

        # 欠陥原子
        defect_mask = np.abs(coord_numbers - 8) > 1
        defect_indices = np.where(defect_mask)[0]
        healthy_indices = np.where(~defect_mask)[0]

        clusters = {}
        cluster_id = 0

        # 欠陥領域の詳細クラスター（各欠陥とその周辺）
        for defect in defect_indices[:100]:  # 最大100個の欠陥クラスター
            nearby = tree.query_ball_point(positions[defect], 5.0)
            clusters[cluster_id] = nearby
            cluster_id += 1

        # 健全領域の空間クラスター
        if len(healthy_indices) > 0:
            n_healthy_clusters = min(50, len(healthy_indices) // 100)

            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_healthy_clusters, random_state=42)
            labels = kmeans.fit_predict(positions[healthy_indices])

            for i in range(n_healthy_clusters):
                cluster_atoms = healthy_indices[labels == i]
                clusters[cluster_id] = cluster_atoms.tolist()
                cluster_id += 1

    elif method == 'fine':
        # 詳細型：小さなクラスターを多数生成
        n_clusters = min(200, n_atoms // 30)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(positions)

        clusters = defaultdict(list)
        for i in range(n_atoms):
            clusters[labels[i]].append(i)

        clusters = dict(clusters)

    else:  # coarse
        # 粗視型：大きなクラスターを少数生成
        n_clusters = max(20, n_atoms // 500)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(positions)

        clusters = defaultdict(list)
        for i in range(n_atoms):
            clusters[labels[i]].append(i)

        clusters = dict(clusters)

    print(f"  Created {len(clusters)} hybrid clusters")
    sizes = [len(v) for v in clusters.values()]
    print(f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")

    return clusters

def main():
    """メイン処理"""
    print("="*60)
    print("💎 SUJ2 Cluster Definition Generator")
    print("="*60)

    # データ読み込み
    print("\n📁 Loading data...")
    trajectory_path = "suj2_fatigue_coarse.npy"
    atom_types_path = "atom_types.npy"

    try:
        trajectory = np.load(trajectory_path)
        atom_types = np.load(atom_types_path)
        print(f"  Trajectory shape: {trajectory.shape}")
        print(f"  Atom types shape: {atom_types.shape}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("  Make sure you're running this script in the suj2_fatigue directory")
        return 1

    # 初期構造を使用
    positions = trajectory[0]
    n_atoms = positions.shape[0]
    print(f"  Using initial structure: {n_atoms} atoms")

    # 各種クラスター生成
    print("\n" + "="*40)

    # 1. 配位数ベース（欠陥検出用）
    coord_clusters = create_coordination_based_clusters(positions)
    with open('clusters_coordination.json', 'w') as f:
        json.dump(coord_clusters, f, indent=2)
    print("  ✅ Saved: clusters_coordination.json")

    # 2. 原子種ベース（相分離解析用）
    if atom_types.dtype.kind in ['U', 'S']:  # 文字列型の場合
        species_clusters = create_atomic_species_clusters(positions, atom_types)
        with open('clusters_species.json', 'w') as f:
            json.dump(species_clusters, f, indent=2)
        print("  ✅ Saved: clusters_species.json")

    # 3. 空間グリッド（局所歪み解析用）
    grid_clusters = create_spatial_grid_clusters(positions, n_per_dim=4)
    # キーをint型に変換（NumPy int64 -> Python int）
    grid_clusters = {int(k): v for k, v in grid_clusters.items()}
    with open('clusters_grid.json', 'w') as f:
        json.dump(grid_clusters, f, indent=2)
    print("  ✅ Saved: clusters_grid.json")

    # 4. 欠陥領域フォーカス（転位・亀裂解析用）
    defect_clusters = create_defect_region_clusters(positions)
    # キーをint型に変換
    defect_clusters = {int(k): v for k, v in defect_clusters.items()}
    with open('clusters_defect_regions.json', 'w') as f:
        json.dump(defect_clusters, f, indent=2)
    print("  ✅ Saved: clusters_defect_regions.json")

    # 5. ハイブリッド（推奨）
    for method in ['balanced', 'fine', 'coarse']:
        hybrid_clusters = create_hybrid_clusters(positions, atom_types, method)
        # キーをint型に変換
        hybrid_clusters = {int(k): v for k, v in hybrid_clusters.items()}
        filename = f'clusters_hybrid_{method}.json'
        with open(filename, 'w') as f:
            json.dump(hybrid_clusters, f, indent=2)
        print(f"  ✅ Saved: {filename}")

    # 推奨設定を標準名で保存
    print("\n🎯 Creating recommended cluster definition...")
    with open('cluster_definition.json', 'w') as f:
        json.dump(hybrid_clusters, f, indent=2)  # balanced版を使用
    print("  ✅ Saved: cluster_definition.json (recommended)")

    print("\n" + "="*60)
    print("✅ Cluster generation complete!")
    print("\nGenerated files:")
    print("  - cluster_definition.json (⭐ RECOMMENDED)")
    print("  - clusters_coordination.json (defect detection)")
    print("  - clusters_species.json (phase analysis)")
    print("  - clusters_grid.json (local strain)")
    print("  - clusters_defect_regions.json (crack/dislocation)")
    print("  - clusters_hybrid_*.json (various granularities)")
    print("\nUsage:")
    print("  python3 ../lambda3_gpu/material_analysis/material_full_analysis.py \\")
    print("    suj2_fatigue_coarse.npy atom_types.npy \\")
    print("    --clusters cluster_definition.json \\")
    print("    -m SUJ2 -o analysis -v")
    print("="*60)

    return 0

if __name__ == "__main__":
    exit(main())
