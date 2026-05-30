from setuptools import setup, find_packages

setup(
    name="multimodal_agent",
    version="1.0.0",
    description="Production-level Multi-modal Generation Agent",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        "console_scripts": [
            "mma-train=training.train:main",
            "mma-sample=sampling.sample:main",
            "mma-eval=evaluation.evaluate:main",
            "mma-agent=agent.multimodal_agent:main",
        ]
    },
)