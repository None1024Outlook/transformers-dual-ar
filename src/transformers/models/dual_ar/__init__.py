from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure

# _import_structure = {
#     "llama": [
#         "DualARModelArgs",
#         "BaseModelArgs",
#         "NaiveModelArgs",
#         "BaseTransformer",
#         "NaiveTransformer",
#         "DualARTransformer"
#     ]
# }

# from .llama import BaseModelArgs
# from .llama import NaiveModelArgs
from .llama import DualARModelArgs
# from .llama import BaseTransformer
# from .llama import NaiveTransformer
from .llama import DualARTransformer
# from .llama import *

# if TYPE_CHECKING:
#     from .llama import *
#     from .lora import *
#     from .lit_module import *
# else:
#     import sys

#     _file = globals()["__file__"]
#     sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
