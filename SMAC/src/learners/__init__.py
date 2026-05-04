from .q_learner import QLearner
from .qplex_learner import DMAQ_qattenLearner as QPlexLearner
from .qplex_decomm_learner import DMAQ_DeCommLearner as QPlexDeCommLearner
from .qplex_learner_teacher import DMAQ_qattenLearner as QPlexLearner_teacher
from .q_learner_teacher import  QLearner as QLearner_teacher
from .q_decomm_learner import QDeCommLearner
from .ptde_learner import PTDELearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["q_learner_teacher"] = QLearner_teacher
REGISTRY["q_decomm_learner"] = QDeCommLearner

REGISTRY["qplex_learner"] = QPlexLearner
REGISTRY["qplex_decomm_learner"] = QPlexDeCommLearner
REGISTRY["qplex_learner_teacher"] = QPlexLearner_teacher

REGISTRY["ptde_learner"] = PTDELearner
