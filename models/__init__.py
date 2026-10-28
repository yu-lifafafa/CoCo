# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-9-24
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from .resnet import *
try:
    from .tresnet import *
except ModuleNotFoundError as error:
    if error.name != "inplace_abn":
        raise
    from .factory import register_unavailable_backbones

    register_unavailable_backbones(
        (
            "tresnet_m",
            "tresnet_l",
            "tresnet_xl",
            "tresnet_lv2",
            "tresnet_m21k",
            "tresnet_l21k",
        ),
        dependency="inplace_abn",
        error=error,
    )
from .timm_models import *
from .mlic import *
from .factory import create_model
