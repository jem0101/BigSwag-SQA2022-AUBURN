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

from nupic.torch.modules import update_boost_strength


class UpdateBoostStrength:
    """
    Update the KWinners boost strength before every epoch.
    """
    def pre_epoch(self):
        super().pre_epoch()
        self.model.apply(update_boost_strength)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["pre_epoch"].append("UpdateBoostStrength")
        return eo
