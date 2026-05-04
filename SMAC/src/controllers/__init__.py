REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .decomm_controller import DeCommMAC
from .ptde_controller import NptdeMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["decomm_mac"] = DeCommMAC
REGISTRY["n_ptde_mac"] = NptdeMAC
