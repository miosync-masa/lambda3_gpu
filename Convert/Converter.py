#!/usr/bin/env python
"""
GROMACS to Lambda³GPU Converter
SUJ2軸受け鋼の疲労解析用
"""

import MDAnalysis as mda
import numpy as np
import json
from pathlib import Path

def convert_gromacs_to_lambda3(
    gro_file='suj2.gro',
    xtc_file='traj.xtc',
    output_dir='lambda3_input'
):
    """GROMACSファイルをLambda³GPU形式に変換"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. トラジェクトリ読み込み
    print("📁 Loading GROMACS trajectory...")
    u = mda.Universe(gro_file, xtc_file)
    
    # 2. 座標データ抽出 → trajectory.npy
    print("🔄 Converting coordinates...")
    coordinates = []
    for ts in u.trajectory:
        coordinates.append(ts.positions)
    
    trajectory = np.array(coordinates, dtype=np.float32)
    print(f"   Shape: {trajectory.shape} (frames, atoms, xyz)")
    
    # nm → Å変換（GROMACSはnm、Lambda³はÅ）
    trajectory *= 10.0
    
    np.save(output_path / 'trajectory.npy', trajectory)
    print(f"   ✅ Saved: trajectory.npy")
    
    # 3. SUJ2原子インデックス作成
    print("🔍 Creating atom indices...")
    
    # Fe原子を選択（SUJ2の主成分）
    fe_atoms = u.select_atoms("name FE*")
    # C原子（炭素鋼成分）
    c_atoms = u.select_atoms("name C*")
    # Cr原子（クロム成分）
    cr_atoms = u.select_atoms("name CR*")
    
    # 全解析対象原子
    steel_indices = np.concatenate([
        fe_atoms.indices,
        c_atoms.indices,
        cr_atoms.indices
    ])
    
    np.save(output_path / 'protein_indices.npy', steel_indices)
    print(f"   Fe: {len(fe_atoms)}, C: {len(c_atoms)}, Cr: {len(cr_atoms)}")
    print(f"   ✅ Saved: protein_indices.npy")
    
    # 4. メタデータ作成
    print("📝 Creating metadata...")
    
    metadata = {
        "system": "SUJ2 bearing steel",
        "n_frames": len(u.trajectory),
        "n_atoms": len(u.atoms),
        "dt_ps": u.trajectory.dt,  # タイムステップ（ps）
        "temperature_K": 300,
        "pressure_MPa": 500,
        "simulation": {
            "software": "GROMACS",
            "force_field": "EAM",
            "timestep_fs": 10,  # 0.01ps = 10fs
            "total_time_ns": len(u.trajectory) * u.trajectory.dt / 1000
        },
        "material": {
            "composition": {
                "Fe": len(fe_atoms),
                "C": len(c_atoms),
                "Cr": len(cr_atoms)
            },
            "crystal_structure": "BCC/FCC mixed",
            "expected_failure": "fatigue crack at grain boundaries"
        },
        "protein": {  # Lambda³は"protein"キーを期待
            "n_residues": len(steel_indices) // 100,  # 仮想的な"残基"
            "sequence": "SUJ2"
        }
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Saved: metadata.json")
    
    # 5. 原子マッピング（オプション）
    print("🔗 Creating atom mapping...")
    
    atom_mapping = {}
    for i, idx in enumerate(steel_indices):
        atom = u.atoms[idx]
        grain_id = i // 1000  # 1000原子ごとに"結晶粒"
        atom_mapping[str(idx)] = {
            "grain": grain_id,
            "element": atom.name[:2],
            "position_type": "bulk" if i % 100 > 10 else "boundary"
        }
    
    with open(output_path / 'atom_mapping.json', 'w') as f:
        json.dump(atom_mapping, f)
    print(f"   ✅ Saved: atom_mapping.json")
    
    print("\n✨ Conversion complete!")
    print(f"📂 Output directory: {output_path}")
    
    return output_path

# 実行例
if __name__ == "__main__":
    # 1000サイクルの疲労シミュレーション後
    convert_gromacs_to_lambda3(
        gro_file='suj2_initial.gro',
        xtc_file='fatigue_1000cycles.xtc',
        output_dir='lambda3_suj2_input'
    )
