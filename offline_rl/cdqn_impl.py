import copy
import d3rlpy
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.optim import Optimizer


from d3rlpy.gpu import Device
from d3rlpy.models.builders import create_discrete_q_function
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch import EnsembleDiscreteQFunction, EnsembleQFunction
from d3rlpy.models.torch.q_functions.ensemble_q_function import _reduce_quantile_ensemble
from d3rlpy.preprocessing import RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, hard_sync, torch_api, train_api
from d3rlpy.algos.torch.base import TorchImplBase
from d3rlpy.algos.torch.utility import DiscreteQFunctionMixin
from d3rlpy.models.torch.q_functions.utility import pick_value_by_action

class CDQNImpl(DiscreteQFunctionMixin, TorchImplBase):
    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _gamma: float
    _n_critics: int
    _use_gpu: Optional[Device]
    _q_func: Optional[EnsembleDiscreteQFunction]
    _targ_q_func: Optional[EnsembleDiscreteQFunction]
    _optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        n_critics: int,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            scaler=scaler,
            action_scaler=None,
            reward_scaler=reward_scaler,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = encoder_factory
        self._q_func_factory = q_func_factory
        self._gamma = gamma
        self._n_critics = n_critics
        self._use_gpu = use_gpu

        # initialized in build
        self._q_func = None
        self._targ_q_func = None
        self._optim = None
       

    def build(self) -> None:
        # setup torch models
        self._build_network()
        
        # setup target network
        self._targ_q_func = copy.deepcopy(self._q_func)
        
        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_optim()
        

    def _build_network(self) -> None:
         self._q_func = create_discrete_q_function(
            self._observation_shape,
            self._action_size,
            self._encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )

    def _build_optim(self) -> None:
        assert self._q_func is not None
        self._optim = self._optim_factory.create(
            self._q_func.parameters(), lr=self._learning_rate
        )

    @train_api
    @torch_api(scaler_targets=["obs_t", "obs_tpn"])
    def update(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._optim is not None

        self._optim.zero_grad()


        q_tpn = self.compute_target(batch) #self.get_next_states(batch) # q_tpn = self.compute_target(batch) # q_tpn = self.compute_next_states(batch) 
        q_cpn = self.compute_next_states(batch)

        loss = max(self.compute_loss(batch, q_tpn), self.compute_loss(batch, q_cpn))

        loss.backward()
        self._optim.step()

        return loss.cpu().detach().numpy()

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_pn: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        assert self._optim is not None
        return self._q_func.compute_error(
            batch.observations,
            batch.actions.long(),
            batch.rewards,
            q_pn,
            batch.terminals,
            self._gamma**batch.n_steps,
        )

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        with torch.no_grad():
            next_actions = self._targ_q_func(batch.next_observations)
            max_action = next_actions.argmax(dim=1)
            print(max_action)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                max_action,
                reduction="min",
            )
            
    def compute_next_states(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._q_func is not None
        with torch.no_grad():
            next_actions = self._q_func(batch.next_observations)
            max_action = next_actions.argmax(dim=1)
            print(max_action)
            return self._q_func.compute_target(
                batch.next_observations,
                max_action,
                reduction="min",
            )
            
    def get_next_states(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._q_func is not None
        with torch.no_grad():
            x = batch.next_observations
            lam: float = 0.75
            reduction: str = "min"
            next_actions = self._q_func(x)
            max_action = next_actions.argmax(dim=1)
            print(max_action)
            values_list: List[torch.Tensor] = []
            for q_func in self._q_func._q_funcs:
                if max_action is None: 
                    next_state = q_func.forward(x)
                else:
                    next_state = pick_value_by_action(q_func.forward(x), max_action, keepdim=True)
                values_list.append(next_state.reshape(1, x.shape[0], -1))
                
            values = torch.cat(values_list, dim=0)
            
            if max_action is None:
                print("actions are None! Please add something to the code.")

            return _reduce_quantile_ensemble(values, reduction, lam=lam)
                
                

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func(x).argmax(dim=1)

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict_best_action(x)

    def update_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        hard_sync(self._targ_q_func, self._q_func)

    @property
    def q_function(self) -> EnsembleQFunction:
        assert self._q_func
        return self._q_func

