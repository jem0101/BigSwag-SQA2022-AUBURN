# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

from nupic.research.frameworks.vernon.mixins.step_based_logging import StepBasedLogging


class LogEveryLearningRate(StepBasedLogging):
    """
    Include the learning rate for every batch in the result dict.

    Adjust config["log_timestep_freq"] to reduce the logging frequency.
    """
    def setup_experiment(self, config):
        super().setup_experiment(config)
        self.lr_history = []
        self.momentum_history = []

    def post_batch(self, batch_idx, **kwargs):
        super().post_batch(batch_idx=batch_idx, **kwargs)

        if self.should_log_batch(batch_idx):
            # Get the lr and momentum from the first param group.
            for param_group in self.optimizer.param_groups:
                lr = param_group["lr"]
                momentum = param_group["momentum"]
                break

            self.lr_history.append(lr)
            self.momentum_history.append(momentum)

    def run_epoch(self):
        result = super().run_epoch()

        result["lr_history"] = self.lr_history
        self.lr_history = []
        result["momentum_history"] = self.momentum_history
        self.momentum_history = []

        return result

    @classmethod
    def expand_result_to_time_series(cls, result, config):
        result_by_timestep = super().expand_result_to_time_series(result,
                                                                  config)

        for t, lr, momentum in zip(cls.get_recorded_timesteps(result, config),
                                   result["lr_history"],
                                   result["momentum_history"]):
            result_by_timestep[t].update(
                learning_rate=lr,
                momentum=momentum,
            )

        return result_by_timestep

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("LogEveryLearningRate: initialize")
        eo["post_batch"].append("LogEveryLearningRate: copy learning rate")
        eo["run_epoch"].append("LogEveryLearningRate: to result dict")
        eo["expand_result_to_time_series"].append(
            "LogEveryLearningRate: learning_rate, momentum"
        )
        return eo
