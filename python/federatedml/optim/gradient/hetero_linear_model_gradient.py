#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import functools

import numpy as np
import scipy.sparse as sp

from federatedml.feature.sparse_vector import SparseVector
from federatedml.statistic import data_overview
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.util import fate_operator
from federatedml.util.fixpoint_solver import FixedPointEncoder


class HeteroGradientBase(object):
    def __init__(self):
        self.use_async = False
        self.use_sample_weight = False
        self.fixed_point_encoder = None

    def compute_gradient_procedure(self, *args):
        raise NotImplementedError("Should not call here")

    def set_total_batch_nums(self, total_batch_nums):
        """
        Use for sqn gradient.
        """
        pass

    def set_use_async(self):
        self.use_async = True

    def set_use_sync(self):
        self.use_async = False

    def set_use_sample_weight(self):
        self.use_sample_weight = True

    def set_fixed_float_precision(self, floating_point_precision):
        if floating_point_precision is not None:
            self.fixed_point_encoder = FixedPointEncoder(2**floating_point_precision)

    @staticmethod
    def __apply_cal_gradient(data, fixed_point_encoder, is_sparse):
        all_g = None
        for key, (feature, d) in data:
            if is_sparse:
                x = np.zeros(feature.get_shape())
                for idx, v in feature.get_all_data():
                    x[idx] = v
                feature = x
            if fixed_point_encoder:
                # g = (feature * 2 ** floating_point_precision).astype("int") * d
                g = fixed_point_encoder.encode(feature) * d
            else:
                g = feature * d
            if all_g is None:
                all_g = g
            else:
                all_g += g
        if all_g is None:
            return all_g
        elif fixed_point_encoder:
            all_g = fixed_point_encoder.decode(all_g)
        return all_g

    def compute_gradient(self, data_instances, fore_gradient, fit_intercept, need_average=True):
        """
        Compute hetero-regression gradient
        Parameters
        ----------
        data_instances: Table, input data
        fore_gradient: Table, fore_gradient
        fit_intercept: bool, if model has intercept or not
        need_average: bool, gradient needs to be averaged or not

        Returns
        ----------
        Table
            the hetero regression model's gradient
        """

        # feature_num = data_overview.get_features_shape(data_instances)
        # data_count = data_instances.count()
        is_sparse = data_overview.is_sparse_data(data_instances)

        LOGGER.debug("Use apply partitions")
        # 把data_instances和fore_gradient进行连接
        feat_join_grad = data_instances.join(fore_gradient,
                                             lambda d, g: (d.features, g))
        # functools.partial 的作用：在原函数的基础上创建一个新的函数，利用原有函数的代码，可以省略掉相同的代码
        # __apply_cal_gradient(data, fixed_point_encoder, is_sparse)中有3个参数
        # 通过 functools.partial 的操作得到一个f函数对象，这个函数就只需要传入一个data参数即可
        # 这样就实现了固定__apply_cal_gradient其中几个参数，一定程度上达到了缩减函数参数个数，
        #
        # 场景：有函数需要传入一个函数对象func，目前已有符合功能的函数A，但A和func的参数signature不同，
        # 可以使用functools.partial 来给A提供某些参数的值来达到参数的“缩减”而创建出一个新的函数对象B，B的参数Signature
        # 符合func的参数Signature即可
        f = functools.partial(self.__apply_cal_gradient,
                              fixed_point_encoder=self.fixed_point_encoder,
                              is_sparse=is_sparse)
        gradient_sum = feat_join_grad.applyPartitions(f)
        gradient_sum = gradient_sum.reduce(lambda x, y: x + y)
        if fit_intercept:
            # bias_grad = np.sum(fore_gradient)
            bias_grad = fore_gradient.reduce(lambda x, y: x + y)
            gradient_sum = np.append(gradient_sum, bias_grad)

        if need_average:
            gradient = gradient_sum / data_instances.count()
        else:
            gradient = gradient_sum

        """
        else:
            LOGGER.debug(f"Original_method")
            feat_join_grad = data_instances.join(fore_gradient,
                                                 lambda d, g: (d.features, g))
            f = functools.partial(self.__compute_partition_gradient,
                                  fit_intercept=fit_intercept,
                                  is_sparse=is_sparse)
            gradient_partition = feat_join_grad.applyPartitions(f)
            gradient_partition = gradient_partition.reduce(lambda x, y: x + y)

            gradient = gradient_partition / data_count
        """
        return gradient


class Guest(HeteroGradientBase):
    def __init__(self):
        super().__init__()
        self.half_d = None
        self.host_forwards = None
        self.forwards = None
        self.aggregated_forwards = None

    def _register_gradient_sync(self, host_forward_transfer, fore_gradient_transfer,
                                guest_gradient_transfer, guest_optim_gradient_transfer):
        self.host_forward_transfer = host_forward_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.unilateral_gradient_transfer = guest_gradient_transfer
        self.unilateral_optim_gradient_transfer = guest_optim_gradient_transfer

    def compute_and_aggregate_forwards(self, data_instances, model_weights,
                                       cipher, batch_index, current_suffix, offset=None):
        raise NotImplementedError("Function should not be called here")

    def compute_half_d(self, data_instances, w, cipher, batch_index, current_suffix):
        raise NotImplementedError("Function should not be called here")

    def _asynchronous_compute_gradient(self, data_instances, model_weights, cipher, current_suffix):
        LOGGER.debug("Called asynchronous gradient")
        LOGGER.debug(f"[guest], half_d before encrypt: {self.half_d}")
        # 这里同态加密, 这里的half_d的数据类型为:<fate_arch.computing.standalone._table.Table>
        encrypted_half_d = cipher.distribute_encrypt(self.half_d)
        LOGGER.debug(f"[guest], half_d after encrypt->encrypted_half_d: {encrypted_half_d}")
        LOGGER.debug(f"[guest], send encrypted_half_d to host: {encrypted_half_d}")
        self.remote_fore_gradient(encrypted_half_d, suffix=current_suffix)  # 把加密后的发送给HOST
        half_g = self.compute_gradient(data_instances, self.half_d, False)  # 计算本地梯度 half_g
        self.host_forwards = self.get_host_forward(suffix=current_suffix)  # 获取由HOST发送的forward
        LOGGER.debug(f"[guest], receive encrypted_forward from host: {self.host_forwards}")
        host_forward = self.host_forwards[0]

        host_half_g = self.compute_gradient(data_instances, host_forward, False)  # 使用host_forward，本地的数据集，计算host_half_g
        unilateral_gradient = half_g + host_half_g  # guest_half_g + host_half_g
        if model_weights.fit_intercept:  # 如果包含截距
            n = data_instances.count()  # 样本数量
            # 把host_forward中的数据进行累加，把half_d中的数据也进行累加，然后把它们的结果相加后再除以样本数量作为截距
            intercept = (host_forward.reduce(lambda x, y: x + y) + self.half_d.reduce(lambda x, y: x + y)) / n
            unilateral_gradient = np.append(unilateral_gradient, intercept)
        return unilateral_gradient

    def _centralized_compute_gradient(self, data_instances, model_weights, cipher, current_suffix, masked_index=None):
        self.host_forwards = self.get_host_forward(suffix=current_suffix)
        fore_gradient = self.half_d

        batch_size = data_instances.count()
        partial_masked_index_enc = None
        if masked_index:
            masked_index = masked_index.mapValues(lambda value: 0)
            masked_index_to_encrypt = masked_index.subtractByKey(self.half_d)
            partial_masked_index_enc = cipher.distribute_encrypt(masked_index_to_encrypt)

        for host_forward in self.host_forwards:
            if self.use_sample_weight:
                # host_forward = host_forward.join(data_instances, lambda h, v: h * v.weight)
                host_forward = data_instances.join(host_forward, lambda v, h: h * v.weight)
            fore_gradient = fore_gradient.join(host_forward, lambda x, y: x + y)

        def _apply_obfuscate(val):
            val.apply_obfuscator()
            return val
        fore_gradient = fore_gradient.mapValues(lambda val: _apply_obfuscate(val) / batch_size)

        if partial_masked_index_enc:
            masked_fore_gradient = partial_masked_index_enc.union(fore_gradient)
            self.remote_fore_gradient(masked_fore_gradient, suffix=current_suffix)
        else:
            self.remote_fore_gradient(fore_gradient, suffix=current_suffix)

        # self.remote_fore_gradient(fore_gradient, suffix=current_suffix)
        unilateral_gradient = self.compute_gradient(data_instances, fore_gradient,
                                                    model_weights.fit_intercept, need_average=False)
        return unilateral_gradient

    def compute_gradient_procedure(self, data_instances, cipher, model_weights, optimizer,
                                   n_iter_, batch_index, offset=None, masked_index=None):
        """
          Linear model gradient procedure
          Step 1: get host forwards which differ from different algorithm
                  For Logistic Regression and Linear Regression: forwards = wx
                  For Poisson Regression, forwards = exp(wx)

          Step 2: Compute self forwards and aggregate host forwards and get d = fore_gradient

          Step 3: Compute unilateral gradient = ∑d*x,

          Step 4: Send unilateral gradients to arbiter and received the optimized and decrypted gradient.
          """
        current_suffix = (n_iter_, batch_index)
        # self.host_forwards = self.get_host_forward(suffix=current_suffix)

        # Compute Guest's partial d
        self.compute_half_d(data_instances, model_weights, cipher,
                            batch_index, current_suffix)  # 会计算self.half_d的值
        if self.use_async:  # 这个值会受batch_size的影响
            unilateral_gradient = self._asynchronous_compute_gradient(data_instances, model_weights,
                                                                      cipher=cipher,
                                                                      current_suffix=current_suffix)
        else:
            unilateral_gradient = self._centralized_compute_gradient(data_instances, model_weights,
                                                                     cipher=cipher,
                                                                     current_suffix=current_suffix,
                                                                     masked_index=masked_index)

        if optimizer is not None:
            LOGGER.debug(f"[guest], unilateral_gradient before add_regular_to_grad: {unilateral_gradient}")
            unilateral_gradient = optimizer.add_regular_to_grad(unilateral_gradient, model_weights)
            LOGGER.debug(f"[guest], unilateral_gradient after add_regular_to_grad: {unilateral_gradient}")

        LOGGER.debug(f"[guest], send unilateral_gradient to arbiter: {unilateral_gradient}")
        optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)
        LOGGER.debug(f"[guest], receive optimized_gradient from arbiter: {optimized_gradient}")
        # LOGGER.debug(f"Before return, optimized_gradient: {optimized_gradient}")
        return optimized_gradient

    def get_host_forward(self, suffix=tuple()):
        host_forward = self.host_forward_transfer.get(idx=-1, suffix=suffix)
        return host_forward

    def remote_fore_gradient(self, fore_gradient, suffix=tuple()):
        self.fore_gradient_transfer.remote(obj=fore_gradient, role=consts.HOST, idx=-1, suffix=suffix)

    def update_gradient(self, unilateral_gradient, suffix=tuple()):
        self.unilateral_gradient_transfer.remote(unilateral_gradient, role=consts.ARBITER, idx=0, suffix=suffix)  # 把单边梯度发送给ARBITER
        optimized_gradient = self.unilateral_optim_gradient_transfer.get(idx=0, suffix=suffix)  # 获取unilateral_optim_gradient
        return optimized_gradient


class Host(HeteroGradientBase):
    def __init__(self):
        super().__init__()
        self.forwards = None
        self.fore_gradient = None

    def _register_gradient_sync(self, host_forward_transfer, fore_gradient_transfer,
                                host_gradient_transfer, host_optim_gradient_transfer):
        self.host_forward_transfer = host_forward_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.unilateral_gradient_transfer = host_gradient_transfer
        self.unilateral_optim_gradient_transfer = host_optim_gradient_transfer

    def compute_forwards(self, data_instances, model_weights):
        raise NotImplementedError("Function should not be called here")

    def compute_unilateral_gradient(self, data_instances, fore_gradient, model_weights, optimizer):
        raise NotImplementedError("Function should not be called here")

    def _asynchronous_compute_gradient(self, data_instances, cipher, current_suffix):
        LOGGER.debug(f"[host], host_forwards before encrypt: {self.forwards}")
        encrypted_forward = cipher.distribute_encrypt(self.forwards)  # 调用加密
        LOGGER.debug(f"[host], host_forwards after encrypt->encrypted_forward: {encrypted_forward}")
        LOGGER.debug(f"[host], send encrypted_forward to guest: {encrypted_forward}")
        self.remote_host_forward(encrypted_forward, suffix=current_suffix)  # 加密后发给GUEST
        half_g = self.compute_gradient(data_instances, self.forwards, False)
        guest_half_d = self.get_fore_gradient(suffix=current_suffix)  # 获取 fore_gradient, 这个就是由GUEST发送给HOST的加密后的half_d
        LOGGER.debug(f"[host], receive encrypted_half_d from guest: {guest_half_d}")
        guest_half_g = self.compute_gradient(data_instances, guest_half_d, False)
        unilateral_gradient = half_g + guest_half_g
        return unilateral_gradient

    def _centralized_compute_gradient(self, data_instances, cipher, current_suffix):
        encrypted_forward = cipher.distribute_encrypt(self.forwards)
        self.remote_host_forward(encrypted_forward, suffix=current_suffix)

        fore_gradient = self.fore_gradient_transfer.get(idx=0, suffix=current_suffix)

        # Host case, never fit-intercept
        unilateral_gradient = self.compute_gradient(data_instances, fore_gradient, False, need_average=False)
        return unilateral_gradient

    def compute_gradient_procedure(self, data_instances, cipher, model_weights,
                                   optimizer,
                                   n_iter_, batch_index):
        """
        Linear model gradient procedure
        Step 1: get host forwards which differ from different algorithm
                For Logistic Regression: forwards = wx


        """
        current_suffix = (n_iter_, batch_index)

        self.forwards = self.compute_forwards(data_instances, model_weights)

        if self.use_async:
            unilateral_gradient = self._asynchronous_compute_gradient(data_instances,
                                                                      cipher,
                                                                      current_suffix)
        else:
            unilateral_gradient = self._centralized_compute_gradient(data_instances,
                                                                     cipher,
                                                                     current_suffix)

        if optimizer is not None:
            LOGGER.debug(f"[host], unilateral_gradient before add_regular_to_grad: {unilateral_gradient}")
            unilateral_gradient = optimizer.add_regular_to_grad(unilateral_gradient, model_weights)
            LOGGER.debug(f"[host], unilateral_gradient after add_regular_to_grad: {unilateral_gradient}")

        LOGGER.debug(f"[host], send unilateral_gradient to arbiter: {unilateral_gradient}")
        optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)
        LOGGER.debug(f"[host], receive optimized_gradient from arbiter: {optimized_gradient}")
        LOGGER.debug(f"Before return compute_gradient_procedure")
        return optimized_gradient

    def compute_sqn_forwards(self, data_instances, delta_s, cipher):
        """
        To compute Hessian matrix, y, s are needed.
        g = (1/N)*∑(0.25 * wx - 0.5 * y) * x
        y = ∇2^F(w_t)s_t = g' * s = (1/N)*∑(0.25 * x * s) * x
        define forward_hess = ∑(0.25 * x * s)
        """
        sqn_forwards = data_instances.mapValues(
            lambda v: cipher.encrypt(fate_operator.vec_dot(v.features, delta_s.coef_) + delta_s.intercept_))
        # forward_sum = sqn_forwards.reduce(reduce_add)
        return sqn_forwards

    def compute_forward_hess(self, data_instances, delta_s, forward_hess):
        """
        To compute Hessian matrix, y, s are needed.
        g = (1/N)*∑(0.25 * wx - 0.5 * y) * x
        y = ∇2^F(w_t)s_t = g' * s = (1/N)*∑(0.25 * x * s) * x
        define forward_hess = (0.25 * x * s)
        """
        hess_vector = self.compute_gradient(data_instances,
                                            forward_hess,
                                            delta_s.fit_intercept)
        return np.array(hess_vector)

    def remote_host_forward(self, host_forward, suffix=tuple()):
        self.host_forward_transfer.remote(obj=host_forward, role=consts.GUEST, idx=0, suffix=suffix)

    def get_fore_gradient(self, suffix=tuple()):
        host_forward = self.fore_gradient_transfer.get(idx=0, suffix=suffix)
        return host_forward

    def update_gradient(self, unilateral_gradient, suffix=tuple()):
        self.unilateral_gradient_transfer.remote(unilateral_gradient, role=consts.ARBITER, idx=0, suffix=suffix)  # 把单边梯度发送给ARBITER
        optimized_gradient = self.unilateral_optim_gradient_transfer.get(idx=0, suffix=suffix)
        return optimized_gradient


class Arbiter(HeteroGradientBase):
    def __init__(self):
        super().__init__()
        self.has_multiple_hosts = False

    def _register_gradient_sync(self, guest_gradient_transfer, host_gradient_transfer,
                                guest_optim_gradient_transfer, host_optim_gradient_transfer):
        self.guest_gradient_transfer = guest_gradient_transfer
        self.host_gradient_transfer = host_gradient_transfer
        self.guest_optim_gradient_transfer = guest_optim_gradient_transfer
        self.host_optim_gradient_transfer = host_optim_gradient_transfer

    def compute_gradient_procedure(self, cipher, optimizer, n_iter_, batch_index):
        """
        Compute gradients.
        Received local_gradients from guest and hosts. Merge and optimize, then separate and remote back.
        Parameters
        ----------
        cipher: Use for encryption

        optimizer: optimizer that get delta gradient of this iter

        n_iter_: int, current iter nums

        batch_index: int

        """
        current_suffix = (n_iter_, batch_index)
        # 接收来自host和guest的 unilateral_gradient
        host_gradients, guest_gradient = self.get_local_gradient(current_suffix)

        if len(host_gradients) > 1:  # 多Host的情况
            self.has_multiple_hosts = True

        host_gradients = [np.array(h) for h in host_gradients]
        guest_gradient = np.array(guest_gradient)

        size_list = [h_g.shape[0] for h_g in host_gradients]
        size_list.append(guest_gradient.shape[0])

        gradient = np.hstack((h for h in host_gradients))  # np.hstack 就是 水平拼接
        gradient = np.hstack((gradient, guest_gradient))

        grad = np.array(cipher.decrypt_list(gradient))  # 解密

        # LOGGER.debug("In arbiter compute_gradient_procedure, before apply grad: {}, size_list: {}".format(
        #     grad, size_list
        # ))

        delta_grad = optimizer.apply_gradients(grad)  # 这里调用优化器子类的apply_gradients方法

        # LOGGER.debug("In arbiter compute_gradient_procedure, delta_grad: {}".format(
        #     delta_grad
        # ))
        separate_optim_gradient = self.separate(delta_grad, size_list)
        # LOGGER.debug("In arbiter compute_gradient_procedure, separated gradient: {}".format(
        #     separate_optim_gradient
        # ))
        host_optim_gradients = separate_optim_gradient[: -1]
        guest_optim_gradient = separate_optim_gradient[-1]
        # 向HOST和GUEST发送optim_gradient
        self.remote_local_gradient(host_optim_gradients, guest_optim_gradient, current_suffix)
        return delta_grad

    @staticmethod
    def separate(value, size_list):
        """
        Separate value in order to several set according size_list
        Parameters
        ----------
        value: list or ndarray, input data
        size_list: list, each set size

        Returns
        ----------
        list
            set after separate
        """
        separate_res = []
        cur = 0
        for size in size_list:
            separate_res.append(value[cur:cur + size])
            cur += size
        return separate_res

    def get_local_gradient(self, suffix=tuple()):
        host_gradients = self.host_gradient_transfer.get(idx=-1, suffix=suffix)
        # LOGGER.info("Get host_gradient from Host")

        guest_gradient = self.guest_gradient_transfer.get(idx=0, suffix=suffix)
        # LOGGER.info("Get guest_gradient from Guest")

        LOGGER.debug(f"[arbiter], receive unilateral_gradient from host: {host_gradients}")
        LOGGER.debug(f"[arbiter], receive unilateral_gradient from guest: {guest_gradient}")

        return host_gradients, guest_gradient

    def remote_local_gradient(self, host_optim_gradients, guest_optim_gradient, suffix=tuple()):
        for idx, host_optim_gradient in enumerate(host_optim_gradients):
            LOGGER.debug(f"[arbiter], send optim_gradient to host: {host_optim_gradient}")
            self.host_optim_gradient_transfer.remote(host_optim_gradient,
                                                     role=consts.HOST,
                                                     idx=idx,
                                                     suffix=suffix)

        LOGGER.debug(f"[arbiter], send optim_gradient to guest: {guest_optim_gradient}")
        self.guest_optim_gradient_transfer.remote(guest_optim_gradient,
                                                  role=consts.GUEST,
                                                  idx=0,
                                                  suffix=suffix)
