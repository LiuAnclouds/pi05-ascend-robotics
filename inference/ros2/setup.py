from glob import glob

from setuptools import find_packages, setup

package_name = "pi05_inference"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy", "PyYAML>=6.0"],
    zip_safe=True,
    maintainer="pi05-ascend-robotics maintainers",
    maintainer_email="maintainer@example.invalid",
    description="Separated ROS2 prompt, video, and Ascend inference nodes.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "prompt_node = pi05_inference.nodes.prompt_node:main",
            "video_node = pi05_inference.nodes.video_node:main",
            "inference_node = pi05_inference.nodes.inference_node:main",
            "robot_node = pi05_inference.nodes.robot_node:main",
        ],
    },
)
