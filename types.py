"""
Lambda3 GPU 統一型定義モジュール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

全プロジェクトで使う型を一箇所で定義！
これで型ヒントの修正が楽になるよ〜！💕

by 環ちゃん
"""

import numpy as np
from typing import Union, Any, TYPE_CHECKING

# CuPyの条件付きインポート
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

# 統一型定義
if TYPE_CHECKING:
    # 型チェック時
    if HAS_CUPY:
        # CuPyがある環境での型チェック
        ArrayType = Union[np.ndarray, cp.ndarray]
    else:
        # CuPyがない環境での型チェック
        ArrayType = Union[np.ndarray, Any]
else:
    # 実行時はシンプルに
    ArrayType = Union[np.ndarray, Any]

# エイリアス（互換性のため）
NDArray = ArrayType

# その他の共通型定義
Numeric = Union[int, float, np.number]
Shape = tuple[int, ...]
DType = Union[np.dtype, str, None]
