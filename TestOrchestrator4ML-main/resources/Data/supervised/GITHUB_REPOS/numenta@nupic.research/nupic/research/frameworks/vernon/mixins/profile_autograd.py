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

import os

import torch.autograd


class ProfileAutograd:
    """
    Use torch's autograd profiler during training.
    """
    def setup_experiment(self, config):
        super().setup_experiment(config)
        # Only profile from rank 0
        self.profile_autograd = self.rank == 0

    def train_epoch(self):
        with torch.autograd.profiler.profile(
                use_cuda=torch.cuda.is_available(),
                enabled=self.profile_autograd) as prof:
            super().train_epoch()

        if self.profile_autograd and prof is not None:
            self.logger.info(prof.key_averages().table(
                sort_by="self_cpu_time_total"))

            prof.export_chrome_trace(os.path.join(
                self.logdir, f"autograd{self.current_epoch}.trace"))

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("ProfileAutograd initialization")
        eo["train_epoch"].insert(0, "ProfileAutograd begin")
        eo["train_epoch"].append("ProfileAutograd end")
        return eo
