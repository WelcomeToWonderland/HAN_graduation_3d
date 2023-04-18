from setuptools import setup, find_packages

"""
tar后缀
tape archive 磁盘存储文档
tarball，tape archive的缩写
系统：linux
对应指令：tar（tar test.txt, tar -d test.txt.tar, d：decompress）
归档文件格式
将多个文件打包为一个文件
但不进行压缩

gz后缀
对应指令：gzip
系统：linux
压缩文件

whl后缀
wheel
python第三方库的打包方式
二进制分发格式
对应安装指令：pip install

编译
包括，预处理，编译，汇编，链接四个部分
最终得到可执行文件（？机器语言）
编译后的程序，可以直接在计算机硬件上运行，不需要借助虚拟机，解释器等

py后缀
python源代码文件
高级编程语言

源代码
source code
未经编译的计算机程序的文本格式

软件包仓库
如PyPI，python package index：官方第三方软件包仓库，python开发者分享自己编写的python软件包
用户可以使用pip install安装包

pip
package installer for python
python包管理器
python软件包的安装，升级，卸载 
从PyPI，获取软件包，完成安装

分发
在计算机领域中，需要将程序，文档资源等，从一个地方转移到另一个地方
分发的方式
1、上传到代码托管平台，如github：git clone
2、打包成源码文件，二进制文件，wheel文件，上传包管理器：pip install
3、打包成镜像文件，上传容器云平台，如docker hub，阿里云平台：拉取镜像

元数据metadata
在软件开发中，描述软件信息的数据，而不是软件本身的信息
通常包含在python项目的，setup.py文件中


项目打包
编写setup.py脚本，运行此脚本，将生成一个可安装的包，如tar.gz文件，whl文件（二进制文件）
tar.gz : python setup.py sdist
sdist：source distribution
sdist命令将项目打包成源代码分发包，通常为tar.gz文件，存储在dist文件夹下

whl : python setup.py bdist_wheel
bdist_wheel命令，生成whl文件，存储在dist文件夹下

setup.py
一个用于打包和发布python项目的脚本文件
通常位于项目根目录
from setuptools import setup, find_packages
setup(
    name='my_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
    ],
)
setup函数，参数为元数据和配置信息
name : python包的名称，而不是项目的名称
find_packages():返回所有包含__init__文件的模块

pip install -e和python setup.py install，python工具包的两张安装方式
pip install -e 项目根目录（项目路径）
e : editable
一种安装python项目的方式，将项目链接到全局环境中，创建一个指向本地包源码目录的符号链接
存在形式：包
要求：完整项目文件+setup.py
好处：支持开发和测试，源码修改时，自动调整包内容，不需要重新安装包

python setup.py install
将包安装到python的lib/site-package目录

site-package
第三方包的安装目录
python解释器，会从site-package中寻找模块和库，同python脚本使用

使用pip install -e安装自己的项目后，项目将被复制到site-package文件夹下
1、直接在项目的src目录下，执行main.py文件，与使用ssh解释器的pycharm执行项目效果一样
2、不在项目中直接执行main.py文件，将执行site-package中复制的项目，也就是输出也会保存在site-package中项目的文件夹中
"""
setup(
    name='my_project',
    version='0.1',
    packages=find_packages(),
)