import hashlib
import numpy as np
from typing import Any, Dict


class MatrixCache:
    def __init__(self):
        self._cache: Dict[str, Any] = {}

    @staticmethod
    def _ensure_numpy_matrix(matrix: Any) -> np.ndarray:
        """确保输入是numpy矩阵，如果是元组则先转换为列表再转矩阵"""
        if isinstance(matrix, tuple):
            matrix = list(matrix)  # 元组转列表
        if not isinstance(matrix, np.ndarray):
            return np.array(matrix)
        return matrix

    def _key_from_matrices(self, matrix1: Any, matrix2: Any) -> str:
        """生成两个矩阵的组合哈希键"""
        # 确保两个矩阵都是numpy数组
        mat1 = self._ensure_numpy_matrix(matrix1)
        mat2 = self._ensure_numpy_matrix(matrix2)
        
        # 生成组合哈希
        h = hashlib.sha256()
        h.update(mat1.tobytes())
        h.update(mat2.tobytes())
        return h.hexdigest()

    def get(self, matrix1: Any, matrix2: Any) -> Any:
        """从缓存获取两个矩阵的计算结果"""
        key = self._key_from_matrices(matrix1, matrix2)
        return self._cache.get(key)

    def set(self, matrix1: Any, matrix2: Any, value: Any) -> None:
        """将两个矩阵的计算结果存入缓存"""
        key = self._key_from_matrices(matrix1, matrix2)
        self._cache[key] = value

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()

    def __len__(self) -> int:
        """返回缓存大小"""
        return len(self._cache)
