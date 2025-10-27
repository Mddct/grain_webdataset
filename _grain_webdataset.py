import json
import glob

import grain
import webdataset as wds
from grain._src.python.dataset import stats as dataset_stats


class _WdsGrainIterator(grain.DatasetIterator):

    def __init__(
        self,
        parent: grain.DatasetIterator,
    ):
        super().__init__(parent)
        self._current_wds_iter = None
        self.samples_processed_in_file = 0

    def _try_get_valid_iter(self):
        if self._current_wds_iter is None:
            try:
                self._current_wds_iter = iter(
                    wds.WebDataset(next(self._parent), shardshuffle=False))
                self.samples_processed_in_file = 0
            except StopIteration:
                return False
        return True

    @dataset_stats.record_next_duration_if_output
    def __next__(self):
        if not self._try_get_valid_iter():
            raise StopIteration
        try:
            tar_line = next(self._current_wds_iter)
            self.samples_processed_in_file += 1
            return tar_line
        except StopIteration:
            self._current_wds_iter = None
            return self.__next__()
        except Exception:
            return self.__next__()

    def get_state(self):
        state = self._parent.get_state()
        return {
            "parent": self._parent.get_state(),
            "samples_processed_in_file": self.samples_processed_in_file,
        }

    def set_state(self, state):
        self._parent.set_state(state['parent'])
        ssamples_to_skip = state.get('samples_processed_in_file')

        if not self._try_get_valid_iter():
            return
        assert self._current_wds_iter is not None
        for _ in range(ssamples_to_skip):
            try:
                _ = next(self._current_wds_iter)
            except StopIteration:
                self._current_wds_iter = None
                break
        self.samples_processed_in_file = ssamples_to_skip

    def __str__(self) -> str:
        return "WdsGrainDatasetIterator(transform=generator)"


class WdsGrainDataset(grain.IterDataset):

    def __init__(self, parent):
        super().__init__(parent)

    def __iter__(self) -> _WdsGrainIterator:
        # Return the wrapped, webdataset-powered iterator
        parent_iter = self._parent.__iter__()
        return _WdsGrainIterator(parent_iter)

if __name__ == '__main__':
  webdataset_path = "*.tar.gz"
  dataset = grain.MapDataset.source(glob.glob(webdataset_path))
  dataset = dataset.to_iter_dataset()
  
  dataset = WdsGrainDataset(dataset)
  for d in dataset:
      print(json.load(d)['text'])


