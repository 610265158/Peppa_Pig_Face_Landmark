from setuptools import setup, find_packages

setup(
    name="Skps",
    version="1.0.0",
    keywords=("Skps"),
    description="Skps sdk",
    long_description="Skps sdk ",
    license="Apache Licence",

    url="",
    author="liangzi",
    author_email="2120140200@mail.nankai.edu.cn",

    packages=find_packages(exclude=['TRAIN*']),
    data_files=['Skps/config/Skps.yml',
                'Skps/pretrained/kps_student.onnx',
                'Skps/pretrained/yolov5n-0.5.onnx'],
    include_package_data=True,
    platforms="any",
    install_requires=['easydict','onnxruntime'],

    scripts=[],
    entry_points={
        'console_scripts': [
            'test = test.help:main'
        ]
    }
)
