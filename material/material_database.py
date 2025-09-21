#!/usr/bin/env python3
"""
Material Database - Unified Material Properties
================================================

統一材料データベース
全てのLambda³ GPU材料解析モジュールで共有される材料パラメータ定義

Version: 1.0.0
Author: 環ちゃん
"""

# ===============================
# 物理定数
# ===============================

K_B = 8.617333e-5  # Boltzmann定数 (eV/K)
AMU_TO_KG = 1.66054e-27  # 原子質量単位→kg
J_TO_EV = 6.242e18  # J→eV変換

# ===============================
# 材料データベース（完全版）
# ===============================

MATERIAL_DATABASE = {
    'SUJ2': {
        'name': 'Bearing Steel (JIS SUJ2)',
        'crystal_structure': 'BCC',
        'lattice_constant': 2.87,      # Å
        'elastic_modulus': 210.0,      # GPa
        'poisson_ratio': 0.30,
        'yield_strength': 1.5,         # GPa
        'ultimate_strength': 2.0,      # GPa
        'fatigue_strength': 0.7,       # GPa
        'fracture_toughness': 30.0,    # MPa√m
        'melting_temp': 1811,          # K
        'density': 7850,               # kg/m³
        'specific_heat': 460,          # J/(kg·K)
        'thermal_expansion': 1.2e-5,   # /K
        'ideal_coordination': 8,       # BCC
        'lindemann_criterion': 0.10,
        'taylor_quinney': 0.90,        # 塑性仕事→熱変換率
        'work_hardening_n': 0.20,      # 加工硬化指数
        'atomic_mass': 55.845,         # Fe (amu)
        # 互換性キー
        'E': 210.0,
        'nu': 0.30,
        'yield': 1.5,
        'ultimate': 2.0,
        'K_IC': 30.0,
        'density_gcm3': 7.85
    },
    'AL7075': {
        'name': 'Aluminum Alloy 7075-T6',
        'crystal_structure': 'FCC',
        'lattice_constant': 4.05,
        'elastic_modulus': 71.7,
        'poisson_ratio': 0.33,
        'yield_strength': 0.503,
        'ultimate_strength': 0.572,
        'fatigue_strength': 0.159,
        'fracture_toughness': 23.0,
        'melting_temp': 933,
        'density': 2810,
        'specific_heat': 900,
        'thermal_expansion': 2.4e-5,
        'ideal_coordination': 12,      # FCC
        'lindemann_criterion': 0.12,
        'taylor_quinney': 0.85,
        'work_hardening_n': 0.15,
        'atomic_mass': 26.982,         # Al (amu)
        # 互換性キー
        'E': 71.7,
        'nu': 0.33,
        'yield': 0.503,
        'ultimate': 0.572,
        'K_IC': 23.0,
        'density_gcm3': 2.81
    },
    'Ti6Al4V': {
        'name': 'Titanium Alloy Ti-6Al-4V',
        'crystal_structure': 'HCP',
        'lattice_constant': 2.95,      # a-axis
        'elastic_modulus': 113.8,
        'poisson_ratio': 0.342,
        'yield_strength': 0.88,
        'ultimate_strength': 0.95,
        'fatigue_strength': 0.51,
        'fracture_toughness': 75.0,
        'melting_temp': 1941,
        'density': 4430,
        'specific_heat': 520,
        'thermal_expansion': 8.6e-6,
        'ideal_coordination': 12,      # HCP
        'lindemann_criterion': 0.11,
        'taylor_quinney': 0.80,
        'work_hardening_n': 0.10,
        'atomic_mass': 47.867,         # Ti (amu)
        # 互換性キー
        'E': 113.8,
        'nu': 0.342,
        'yield': 0.88,
        'ultimate': 0.95,
        'K_IC': 75.0,
        'density_gcm3': 4.43
    },
    'SS316L': {
        'name': 'Stainless Steel 316L',
        'crystal_structure': 'FCC',
        'lattice_constant': 3.58,
        'elastic_modulus': 193.0,
        'poisson_ratio': 0.30,
        'yield_strength': 0.205,
        'ultimate_strength': 0.515,
        'fatigue_strength': 0.24,
        'fracture_toughness': 112.0,
        'melting_temp': 1673,
        'density': 8000,
        'specific_heat': 500,
        'thermal_expansion': 1.6e-5,
        'ideal_coordination': 12,
        'lindemann_criterion': 0.12,
        'taylor_quinney': 0.85,
        'work_hardening_n': 0.45,
        'atomic_mass': 55.845,         # Fe主体 (amu)
        # 互換性キー
        'E': 193.0,
        'nu': 0.30,
        'yield': 0.205,
        'ultimate': 0.515,
        'K_IC': 112.0,
        'density_gcm3': 8.0
    }
}

# ===============================
# ヘルパー関数
# ===============================

def get_material_parameters(material_type: str) -> dict:
    """
    材料パラメータを取得
    
    Parameters
    ----------
    material_type : str
        材料タイプ ('SUJ2', 'AL7075', 'Ti6Al4V', 'SS316L')
    
    Returns
    -------
    dict
        材料パラメータ辞書
    """
    if material_type not in MATERIAL_DATABASE:
        import warnings
        warnings.warn(f"Unknown material {material_type}, using SUJ2")
        material_type = 'SUJ2'
    
    return MATERIAL_DATABASE[material_type].copy()

def get_material_list() -> list:
    """利用可能な材料リストを取得"""
    return list(MATERIAL_DATABASE.keys())

def get_material_info(material_type: str) -> str:
    """材料の詳細情報を文字列で取得"""
    if material_type not in MATERIAL_DATABASE:
        return f"Unknown material: {material_type}"
    
    mat = MATERIAL_DATABASE[material_type]
    info = f"""
Material: {mat['name']}
Crystal Structure: {mat['crystal_structure']}
Lattice Constant: {mat['lattice_constant']} Å
Elastic Modulus: {mat['elastic_modulus']} GPa
Yield Strength: {mat['yield_strength']} GPa
Ultimate Strength: {mat['ultimate_strength']} GPa
Fatigue Strength: {mat['fatigue_strength']} GPa
Fracture Toughness: {mat['fracture_toughness']} MPa√m
Melting Temperature: {mat['melting_temp']} K
Density: {mat['density']} kg/m³
"""
    return info.strip()
