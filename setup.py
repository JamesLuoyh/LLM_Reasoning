import setuptools

setuptools.setup(
    name="llm_reasoning",
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src')
)
