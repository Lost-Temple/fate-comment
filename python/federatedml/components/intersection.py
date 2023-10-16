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

from .components import ComponentMeta


intersection_cpn_meta = ComponentMeta("Intersection")

# 纵向的隐私求交
@intersection_cpn_meta.bind_param
def intersection_param():
    from federatedml.param.intersect_param import IntersectParam

    return IntersectParam


@intersection_cpn_meta.bind_runner.on_guest
def intersection_guest_runner():
    from federatedml.statistic.intersect.intersect_model import IntersectGuest

    return IntersectGuest


@intersection_cpn_meta.bind_runner.on_host
def intersection_host_runner():
    from federatedml.statistic.intersect.intersect_model import IntersectHost

    return IntersectHost
