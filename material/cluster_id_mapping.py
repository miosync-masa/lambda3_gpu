#!/usr/bin/env python3
"""
Cluster ID Mapping Utility for Material Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

細分化されたクラスターID（0, 1001, 1002...）と
配列インデックス（0, 1, 2...）の変換を管理

Version: 1.0
by 環ちゃん - Cluster Mapping Edition
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Iterator
import logging

logger = logging.getLogger('lambda3_gpu.material.cluster_id_mapping')


class ClusterIDMapper:
    """
    クラスターIDと配列インデックスのマッピング管理
    
    細分化後のID体系:
    - 0: 健全領域
    - 1001-1xxx: クラスター1の細分化
    - 2001-2xxx: クラスター2の細分化
    - 3001-3xxx: クラスター3の細分化
    """
    
    def __init__(self, cluster_atoms: Dict[int, List[int]]):
        """
        Parameters
        ----------
        cluster_atoms : Dict[int, List[int]]
            クラスターID → 原子IDリストのマッピング
        """
        # ソート済みクラスターID（0, 1001, 1002, ..., 2001, ...）
        self.cluster_ids = sorted(cluster_atoms.keys())
        self.n_clusters = len(self.cluster_ids)
        
        # 双方向マッピング作成
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.cluster_ids)}
        self.idx_to_id = {idx: cid for cid, idx in self.id_to_idx.items()}
        
        # 元のクラスター情報も保持
        self.cluster_atoms = cluster_atoms
        
        # 統計情報
        self._analyze_structure()
        
        logger.info(f"ClusterIDMapper initialized:")
        logger.info(f"  Total clusters: {self.n_clusters}")
        logger.info(f"  Healthy cluster (0): {0 in self.id_to_idx}")
        logger.info(f"  Subdivided clusters: {self.n_subdivided}")
    
    def _analyze_structure(self):
        """クラスター構造を解析"""
        self.n_healthy = 1 if 0 in self.id_to_idx else 0
        self.n_subdivided = self.n_clusters - self.n_healthy
        
        # 各元クラスターの細分化数をカウント
        self.subdivision_counts = {}
        for cid in self.cluster_ids:
            if cid == 0:
                continue
            parent_id = cid // 1000  # 1001 → 1, 2001 → 2
            if parent_id not in self.subdivision_counts:
                self.subdivision_counts[parent_id] = 0
            self.subdivision_counts[parent_id] += 1
    
    # ========================================
    # 基本変換メソッド
    # ========================================
    
    def to_idx(self, cluster_id: int) -> int:
        """
        クラスターID → 配列インデックス
        
        Examples
        --------
        >>> mapper.to_idx(1001)  # → 1
        >>> mapper.to_idx(2003)  # → 34
        """
        if cluster_id not in self.id_to_idx:
            logger.warning(f"Unknown cluster ID: {cluster_id}, returning 0")
            return 0
        return self.id_to_idx[cluster_id]
    
    def to_id(self, idx: int) -> int:
        """
        配列インデックス → クラスターID
        
        Examples
        --------
        >>> mapper.to_id(0)   # → 0
        >>> mapper.to_id(1)   # → 1001
        >>> mapper.to_id(34)  # → 2003
        """
        if idx not in self.idx_to_id:
            logger.warning(f"Invalid index: {idx}, returning idx")
            return idx
        return self.idx_to_id[idx]
    
    # ========================================
    # バッチ変換
    # ========================================
    
    def ids_to_indices(self, cluster_ids: List[int]) -> np.ndarray:
        """複数のクラスターIDを一度に変換"""
        return np.array([self.to_idx(cid) for cid in cluster_ids], dtype=np.int32)
    
    def indices_to_ids(self, indices: np.ndarray) -> List[int]:
        """複数のインデックスを一度に変換"""
        return [self.to_id(int(idx)) for idx in indices]
    
    # ========================================
    # イテレータ
    # ========================================
    
    def iterate_with_idx(self) -> Iterator[Tuple[int, int]]:
        """
        (cluster_id, index)のペアでイテレート
        
        Yields
        ------
        cluster_id, idx : int, int
            クラスターIDと対応するインデックス
        """
        for idx, cid in self.idx_to_id.items():
            yield cid, idx
    
    def iterate_ids(self) -> Iterator[int]:
        """クラスターIDのみでイテレート"""
        return iter(self.cluster_ids)
    
    def iterate_indices(self) -> Iterator[int]:
        """インデックスのみでイテレート（0からn_clusters-1）"""
        return iter(range(self.n_clusters))
    
    # ========================================
    # 配列操作ヘルパー
    # ========================================
    
    def remap_array(self, array: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        クラスターID次元を持つ配列を再マップ
        
        Parameters
        ----------
        array : np.ndarray
            元の配列（クラスターID次元を持つ）
        axis : int
            クラスター次元の軸
        
        Returns
        -------
        np.ndarray
            インデックスベースに再配置された配列
        """
        # 実装例（簡易版）
        if axis == -1:
            axis = array.ndim - 1
        
        # 新しい形状
        new_shape = list(array.shape)
        new_shape[axis] = self.n_clusters
        new_array = np.zeros(new_shape, dtype=array.dtype)
        
        # データコピー
        for cid, idx in self.iterate_with_idx():
            if cid < array.shape[axis]:
                # スライス作成
                slices = [slice(None)] * array.ndim
                slices[axis] = cid
                src_slice = tuple(slices)
                
                slices[axis] = idx
                dst_slice = tuple(slices)
                
                new_array[dst_slice] = array[src_slice]
        
        return new_array
    
    # ========================================
    # クエリメソッド
    # ========================================
    
    def get_parent_cluster(self, cluster_id: int) -> int:
        """
        細分化クラスターの親クラスターIDを取得
        
        Examples
        --------
        >>> mapper.get_parent_cluster(1001)  # → 1
        >>> mapper.get_parent_cluster(2003)  # → 2
        """
        if cluster_id == 0:
            return 0
        return cluster_id // 1000
    
    def get_subdivision_id(self, cluster_id: int) -> int:
        """
        細分化内でのサブIDを取得
        
        Examples
        --------
        >>> mapper.get_subdivision_id(1001)  # → 1
        >>> mapper.get_subdivision_id(2003)  # → 3
        """
        if cluster_id == 0:
            return 0
        return cluster_id % 1000
    
    def get_sibling_clusters(self, cluster_id: int) -> List[int]:
        """同じ親クラスターから分割された兄弟クラスターを取得"""
        if cluster_id == 0:
            return [0]
        
        parent = self.get_parent_cluster(cluster_id)
        siblings = []
        
        for cid in self.cluster_ids:
            if cid != 0 and self.get_parent_cluster(cid) == parent:
                siblings.append(cid)
        
        return siblings
    
    # ========================================
    # 検証メソッド
    # ========================================
    
    def validate_cluster_id(self, cluster_id: int) -> bool:
        """クラスターIDが有効か確認"""
        return cluster_id in self.id_to_idx
    
    def validate_index(self, idx: int) -> bool:
        """インデックスが有効か確認"""
        return 0 <= idx < self.n_clusters
    
    # ========================================
    # デバッグ用
    # ========================================
    
    def print_mapping(self, limit: int = 10):
        """マッピング情報を表示"""
        print(f"\n=== Cluster ID Mapping ===")
        print(f"Total clusters: {self.n_clusters}")
        print(f"\nFirst {min(limit, self.n_clusters)} mappings:")
        print(f"{'Index':<10} {'Cluster ID':<15} {'Parent':<10} {'Atoms':<10}")
        print("-" * 50)
        
        for i, (idx, cid) in enumerate(self.idx_to_id.items()):
            if i >= limit:
                break
            parent = self.get_parent_cluster(cid)
            n_atoms = len(self.cluster_atoms.get(cid, []))
            print(f"{idx:<10} {cid:<15} {parent:<10} {n_atoms:<10}")
        
        if self.n_clusters > limit:
            print(f"... and {self.n_clusters - limit} more")
    
    def get_statistics(self) -> Dict:
        """統計情報を取得"""
        return {
            'n_clusters': self.n_clusters,
            'n_healthy': self.n_healthy,
            'n_subdivided': self.n_subdivided,
            'subdivision_counts': self.subdivision_counts,
            'cluster_ids_range': (min(self.cluster_ids), max(self.cluster_ids))
        }


# ========================================
# ユーティリティ関数
# ========================================

def create_mapper_from_json(json_path: str) -> ClusterIDMapper:
    """JSONファイルからマッパーを作成"""
    import json
    with open(json_path, 'r') as f:
        cluster_atoms_raw = json.load(f)
    
    # 整数キーに変換
    cluster_atoms = {}
    for k, v in cluster_atoms_raw.items():
        cid = int(k)
        if isinstance(v, dict) and 'atom_ids' in v:
            cluster_atoms[cid] = [int(a) for a in v['atom_ids']]
        elif isinstance(v, list):
            cluster_atoms[cid] = [int(a) for a in v]
    
    return ClusterIDMapper(cluster_atoms)


# テスト用
if __name__ == "__main__":
    # テストデータ
    test_clusters = {
        0: list(range(64000)),  # 健全領域
        1001: list(range(64000, 64100)),
        1002: list(range(64100, 64200)),
        2001: list(range(64200, 64300)),
        2002: list(range(64300, 64400)),
        3001: list(range(64400, 64500))
    }
    
    mapper = ClusterIDMapper(test_clusters)
    mapper.print_mapping()
    
    # テスト
    print("\n=== Tests ===")
    print(f"to_idx(1001): {mapper.to_idx(1001)}")
    print(f"to_id(1): {mapper.to_id(1)}")
    print(f"get_parent_cluster(2002): {mapper.get_parent_cluster(2002)}")
    print(f"get_sibling_clusters(1001): {mapper.get_sibling_clusters(1001)}")
