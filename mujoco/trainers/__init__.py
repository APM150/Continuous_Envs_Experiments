# register trainer here
from trainers.base_trainer import                                   BaseTrainer

# single model trainers
from trainers.single_model_trainers.sac import                      SAC
from trainers.single_model_trainers.ddpg import                     DDPG

# ensemble model trainers
from trainers.ensemble_model_trainers.kfold_sac import              Kfold_SAC
from trainers.ensemble_model_trainers.mean_sac import               Mean_SAC
from trainers.ensemble_model_trainers.mean_ddpg import              Mean_DDPG
from trainers.ensemble_model_trainers.mean_dspg import              Mean_DSPG
