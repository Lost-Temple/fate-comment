#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


from pyspark import StorageLevel


# noinspection PyUnresolvedReferences
def materialize(rdd):
    rdd.persist(get_storage_level())  # 这里会把rdd持久化，并指定了存储级别，但这里并没有真正的做rdd的持久化
    rdd.count()  # rdd的第一次计算会触发rdd的持久化，这里调用rdd.count()的作用就是为了触发rdd的持久化
    return rdd


def unmaterialize(rdd):
    rdd.unpersist()  # 会释放持久化所占用的内存和磁盘空间


# noinspection PyUnresolvedReferences
def get_storage_level():
    return StorageLevel.MEMORY_AND_DISK
