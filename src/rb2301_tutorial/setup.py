from setuptools import find_packages, setup

package_name = 'rb2301_tutorial'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(
        include=['rclpy'],
        exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vkliu',
    maintainer_email='vkliu@uw.edu',
    description='Week 3 NUS RB2301 - Tutorial Project',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tut = rb2301_tutorial.tutorial:main',
            'fake = rb2301_tutorial.fake:main',
            'logger = rb2301_tutorial.logger:main',
            'recorder = rb2301_tutorial.recorder:main',
            'pubs = rb2301_tutorial.publishers:main',
            'subs = rb2301_tutorial.subscribers:main'
        ],
    },
)
