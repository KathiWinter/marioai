from d3rlpy.dataset import MDPDataset
from cdqn import CDQN
from constants import DATAPATH


dataset = MDPDataset.load(DATAPATH)



cdqn = CDQN()
cdqn.build_with_dataset(dataset)
#to get specific model (not the latest), change this file path
cdqn.load_model('d3rlpy_logs\DQN_20220905113722\model_85790.pt')
cdqn.save_policy('d3rlpy_logs\DQN_20220905113722\policy.pt')
