"""Setup configuration for stock-pricer package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from __init__.py
version = {}
with open("stock_pricer/__init__.py") as fp:
    exec(fp.read(), version)
    
setup(
    name="stock-pricer",
    version=version["__version__"],
    author="quinn",
    author_email="your.email@example.com",
    description="A script to fetch & price stocks with trend detection and Monte Carlo simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stock_pricer",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/stock_pricer/issues",
        "Documentation": "https://github.com/yourusername/stock_pricer/blob/main/README.md",
        "Source Code": "https://github.com/yourusername/stock_pricer",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "yfinance>=0.2.28",
        "matplotlib>=3.7.0",
        "plotly>=5.17.0",
        "streamlit>=1.28.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pylint>=3.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.5.0",
            "bandit>=1.7.5",
            "safety>=2.3.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "stock-pricer=stock_pricer.fetch_stocks:main",
            "stock-pricer-web=stock_pricer.streamlit_app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)