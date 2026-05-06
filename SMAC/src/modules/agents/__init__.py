REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent
from .atten_rnn_agent import ATTRNNAgent
REGISTRY["att_rnn"] = ATTRNNAgent
from .decomm_rnn_agent import DecommRNNAgent
REGISTRY["decomm_rnn"] = DecommRNNAgent
from .ptde_rnn_agent import PTDERNNAgent
REGISTRY["ptde_rnn"] = PTDERNNAgent
