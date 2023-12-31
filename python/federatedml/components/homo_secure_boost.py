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
#  横向联邦学习的SecureBoost, 以下信息不一定准确：
#  它是采用同态加密来保证数据的安全，整个过程以符合同态加密的安全。
#  1.数据对齐：它是一种无损的联邦学习框架，在不泄露数据标识的情况下寻求数据交集。
#  2.构造Boost树

from .components import ComponentMeta

homo_secure_boost_cpn_meta = ComponentMeta("HomoSecureBoost", "HomoSecureboost")


@homo_secure_boost_cpn_meta.bind_param
def homo_secure_boost_param():
    from federatedml.param.boosting_param import HomoSecureBoostParam

    return HomoSecureBoostParam


@homo_secure_boost_cpn_meta.bind_runner.on_guest.on_host
def homo_secure_boost_runner_client():
    from federatedml.ensemble import (HomoSecureBoostingTreeClient)

    return HomoSecureBoostingTreeClient


@homo_secure_boost_cpn_meta.bind_runner.on_arbiter
def homo_secure_boost_runner_arbiter():
    from federatedml.ensemble import (HomoSecureBoostingTreeArbiter)

    return HomoSecureBoostingTreeArbiter
