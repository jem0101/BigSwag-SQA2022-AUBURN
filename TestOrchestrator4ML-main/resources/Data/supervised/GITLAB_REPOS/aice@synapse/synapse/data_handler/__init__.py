"""
Copyright (C) 2019  Syed Hasibur Rahman

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# region : Imports
# endregion : Imports


class BaseIterator:
    """
    Base Iterator class
    """
    def __init__(self, dataset, *args, **kwargs):
        self.dataset = dataset

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def max_epoch(self):
        return self.dataset.total_epoch

    @property
    def classes(self):
        return self.dataset.classes.names

    @property
    def num_classes(self):
        return self.dataset.num_classes

    @property
    def num_max_boxes(self):
        return self.dataset.num_max_boxes

    @property
    def label_colors(self):
        return self.dataset.label_colors
