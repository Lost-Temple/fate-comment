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


from federatedml.secureprotol.encrypt import PaillierEncrypt, IpclPaillierEncrypt
from federatedml.util import consts


class Arbiter(object):
    # noinspection PyAttributeOutsideInit
    def _register_paillier_keygen(self, pubkey_transfer):
        self._pubkey_transfer = pubkey_transfer

    def paillier_keygen(self, method, key_length, suffix=tuple()):
        if method == consts.PAILLIER:
            cipher = PaillierEncrypt()
        elif method == consts.PAILLIER_IPCL:
            cipher = IpclPaillierEncrypt()
        else:
            raise ValueError(f"Unsupported encryption method: {method}")

        cipher.generate_key(key_length)
        pub_key = cipher.get_public_key()
        self._pubkey_transfer.remote(obj=pub_key, role=consts.HOST, idx=-1, suffix=suffix)  # 把公钥发给HOST
        self._pubkey_transfer.remote(obj=pub_key, role=consts.GUEST, idx=-1, suffix=suffix)  # 把公钥发给GUEST
        return cipher


class _Client(object):
    # noinspection PyAttributeOutsideInit
    def _register_paillier_keygen(self, pubkey_transfer):  # 生成公钥
        self._pubkey_transfer = pubkey_transfer

    def gen_paillier_cipher_operator(self, suffix=tuple(), method=consts.PAILLIER):
        pubkey = self._pubkey_transfer.get(idx=0, suffix=suffix)

        if method == consts.PAILLIER:  # 帕耶同态加密算法
            cipher = PaillierEncrypt()
        elif method == consts.PAILLIER_IPCL:  # intel芯片加速的帕耶同态加密算法
            cipher = IpclPaillierEncrypt()
        else:
            raise ValueError(f"Unsupported encryption method: {method}")

        cipher.set_public_key(pubkey)
        return cipher


Host = _Client
Guest = _Client
