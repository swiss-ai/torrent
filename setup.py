from setuptools import setup, find_packages

setup(
    name="torrent",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "torrent": ["models/*.yaml", "template.jinja"]
    },
    install_requires=[
        "datasets",
        "sglang",
        "jinja2", 
        "dacite",
        "omegaconf",
        "redislite",
        "prettytable",
        "importlib_resources; python_version<'3.9'"
    ],
    python_requires=">=3.8",
)
