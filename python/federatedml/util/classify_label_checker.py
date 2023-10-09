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
#
################################################################################
#
#
################################################################################

# =============================================================================
# Lable Checker
# =============================================================================

from federatedml.util import consts


class ClassifyLabelChecker(object):
    def __init__(self):
        pass

    # 用于在分类任务中检查标签。它检查不同标签的数量是否超过了consts.MAX_CLASSNUM，
    # 并获取所有不同的标签。该方法接受一个名为data_inst的参数，该参数是在federatedml/feature/instance.py中定义的数据实例格式的表格。
    # 该方法返回两个值：num_class和labels。其中，num_class是不同标签的数量，而labels是不同标签的列表
    @staticmethod
    def validate_label(data_inst):
        """
        Label Checker in classification task.
            Check whether the distinct labels is no more than MAX_CLASSNUM which define in consts,
            also get all distinct labels

        Parameters
        ----------
        data_inst : Table,
                    values are data instance format define in federatedml/feature/instance.py

        Returns
        -------
        num_class : int, the number of distinct labels

        labels : list, the distince labels

        """
        class_set = data_inst.applyPartitions(ClassifyLabelChecker.get_all_class).reduce(lambda x, y: x | y)

        num_class = len(class_set)
        if len(class_set) > consts.MAX_CLASSNUM:
            raise ValueError("In Classfy Proble, max dif classes should no more than %d" % (consts.MAX_CLASSNUM))

        return num_class, list(class_set)

    @staticmethod
    def get_all_class(kv_iterator):
        class_set = set()
        for _, inst in kv_iterator:
            class_set.add(inst.label)

        if len(class_set) > consts.MAX_CLASSNUM:
            raise ValueError("In Classify Task, max dif classes should no more than %d" % (consts.MAX_CLASSNUM))

        return class_set


class RegressionLabelChecker(object):
    @staticmethod
    def validate_label(data_inst):
        """
        Label Checker in regression task.
            Check if all labels is a float type.

        Parameters
        ----------
        data_inst : Table,
                    values are data instance format define in federatedml/feature/instance.py

        """
        data_inst.mapValues(RegressionLabelChecker.test_numeric_data)

    @staticmethod
    def test_numeric_data(value):
        try:
            label = float(value.label)
        except BaseException:
            raise ValueError("In Regression Task, all label should be numeric!!")
