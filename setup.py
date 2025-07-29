"""
Lambda³ GPU - Setup Script
High-performance GPU implementation of Lambda³ framework
"""

from setuptools import setup, find_packages
import os

# バージョン情報
VERSION = '1.0.0'

# READMEを読み込む
with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# 依存関係
install_requires = [
    'numpy>=1.20.0',
    'cupy-cuda11x>=10.0.0',  # CUDA 11.x用、環境に応じて変更
    'scipy>=1.7.0',
    'scikit-learn>=1.0.0',
    'numba>=0.54.0',
    'matplotlib>=3.4.0',
    'seaborn>=0.11.0',
    'plotly>=5.0.0',
    'pandas>=1.3.0',
    'psutil>=5.8.0',
    'GPUtil>=1.4.0',
    'networkx>=2.6.0',
    'joblib>=1.0.0'
]

# 開発用依存関係
extras_require = {
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.0.0',
        'black>=21.0',
        'flake8>=3.9.0',
        'mypy>=0.910',
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0'
    ],
    'viz': [
        'jupyterlab>=3.0.0',
        'ipywidgets>=7.6.0',
        'notebook>=6.4.0'
    ]
}

# CUDAバージョンの自動検出
def get_cuda_version():
    """インストールされているCUDAバージョンを検出"""
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            # バージョン番号を抽出
            import re
            match = re.search(r'release (\d+)\.(\d+)', output)
            if match:
                major, minor = match.groups()
                return f"{major}{minor}"
    except:
        pass
    return None

# CUDAバージョンに応じてCuPyを選択
cuda_version = get_cuda_version()
if cuda_version:
    if cuda_version.startswith('11'):
        install_requires[1] = 'cupy-cuda11x>=10.0.0'
    elif cuda_version.startswith('12'):
        install_requires[1] = 'cupy-cuda12x>=10.0.0'
    else:
        print(f"Warning: CUDA {cuda_version} detected. Using default CuPy.")
else:
    print("Warning: CUDA not detected. GPU features may not work.")

setup(
    name='lambda3-gpu',
    version=VERSION,
    author='Lambda³ Project Team',
    author_email='lambda3@example.com',
    description='GPU-accelerated Lambda³ framework for MD trajectory analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lambda3-project/lambda3-gpu',
    project_urls={
        'Bug Reports': 'https://github.com/lambda3-project/lambda3-gpu/issues',
        'Source': 'https://github.com/lambda3-project/lambda3-gpu',
        'Documentation': 'https://lambda3-gpu.readthedocs.io',
    },
    
    # パッケージ設定
    packages=find_packages(exclude=['tests', 'docs', 'examples']),
    package_dir={'lambda3_gpu': 'lambda3_gpu'},
    
    # Pythonバージョン
    python_requires='>=3.8',
    
    # 依存関係
    install_requires=install_requires,
    extras_require=extras_require,
    
    # エントリーポイント
    entry_points={
        'console_scripts': [
            'lambda3-benchmark=lambda3_gpu.benchmarks.performance_tests:main',
            'lambda3-quick-test=lambda3_gpu.benchmarks.performance_tests:run_quick_benchmark',
        ],
    },
    
    # 分類
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: CUDA',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Environment :: GPU :: NVIDIA CUDA :: 11.0',
        'Environment :: GPU :: NVIDIA CUDA :: 12.0',
    ],
    
    # キーワード
    keywords='molecular-dynamics gpu cuda lambda3 trajectory-analysis protein-folding',
    
    # データファイル
    include_package_data=True,
    package_data={
        'lambda3_gpu': [
            'README.md',
            'LICENSE',
            'requirements.txt'
        ],
    },
    
    # テスト
    test_suite='tests',
    
    # プラットフォーム
    platforms=['Linux', 'Windows'],
    
    # ライセンス
    license='MIT',
)

# インストール後メッセージ
print("""
╔═══════════════════════════════════════════════════════════════╗
║                    Lambda³ GPU Installation                    ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  🎉 Installation complete!                                     ║
║                                                               ║
║  Quick test:                                                  ║
║    $ python -c "from lambda3_gpu import MDLambda3DetectorGPU" ║
║    $ lambda3-quick-test                                      ║
║                                                               ║
║  Full benchmark:                                              ║
║    $ lambda3-benchmark                                        ║
║                                                               ║
║  Documentation:                                               ║
║    https://lambda3-gpu.readthedocs.io                       ║
║                                                               ║
║  NO TIME, NO PHYSICS, ONLY STRUCTURE! 🌌                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")
