安装gmpy2时会缺少一些包，在centos下：
```shell
sudo yum install gmp-devel mpfr-devel
sudo yum install -y libmpc-devel
```
```shell
pip install gmpy2==2.0.8
```
# fate中依赖安装
```shell
cd ${FATE_PROJECT_BASE}/python
pip install -r requirements.txt
```
---
# fate-flow中依赖安装
```shell
pip install grpcio==1.46.3
pip install grpcio-tools==1.46.3
pip install Werkzeug==2.3.7
pip install filelock==3.3.1
pip install cachetools==3.0.0
pip install ruamel-yaml==0.16.10
pip install python-python-dotenv==0.13.0
pip install peewee==3.9.3
pip install apsw==3.9.2.post1
pip install protobuf==3.19.0
pip install Flask==2.0.3
pip install requests==2.25.1
pip install psutil
pip install qcloud-cos-python3
pip3 install -U cos-python-sdk-v5 -i https://mirrors.cloud.tencent.com/pypi/simple
pip install beautifultable==1.0.0
pip install numpy==1.23.1
pip install casbin==1.16.6
pip install pip install casbin-sqlalchemy-adapter==0.4.2
pip install PyMySQL==0.9.3
pip install kazoo==2.6.1
pip install shortuuid==1.0.9
pip install requests_toolbelt==0.9.1
pip install cloudpickle==2.1.0
pip install lmdb==1.3.0
```

