"""Setup script for BizSimAgent."""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="bizsim-agent",
        version="0.1.0",
        packages=find_packages(),
        install_requires=[
            "typer>=0.9.0",
            "pydantic>=2.0.0",
            "rich>=13.0.0",
            "numpy>=1.24.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        entry_points={
            "console_scripts": [
                "bizsim=bizsim.cli:app",
            ],
        },
        python_requires=">=3.9",
    )
