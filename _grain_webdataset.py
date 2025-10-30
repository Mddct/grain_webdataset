import grain
import webdataset as wds
from grain._src.python.dataset import stats as dataset_stats

audio_formats = [
    'wav',
    'flac',
    'mp3',
    'm4a',
    'ogg',
    'aac',
    'aiff',
    'au',
    'opus',
    'amr',
    'mp4',
    'avi',
    'wmv',
    'mpeg',
    'mkv',
    'mov',
]


def rename_audio_to_wav(sample):
    current_audio_key = None
    for key in sample:
        if key in audio_formats:
            current_audio_key = key
            break

    if current_audio_key and current_audio_key != 'wav':
        sample['wav'] = sample.pop(current_audio_key)

    return sample


def no_split(src):
    """ NOTE: grain is responsible for splitting data
    """
    yield from src


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
                    wds.WebDataset(next(self._parent),
                                   shardshuffle=False,
                                   nodesplitter=no_split,
                                   workersplitter=no_split))
                self.samples_processed_in_file = 0
            except StopIteration:
                return False
        return True

    @dataset_stats.record_next_duration_if_output
    def __next__(self):
        while True:
            if not self._try_get_valid_iter():
                raise StopIteration
            try:
                elem_dict = next(self._current_wds_iter)
                elem_dict = rename_audio_to_wav(elem_dict)
                self.samples_processed_in_file += 1
                return elem_dict

            except StopIteration:
                self._current_wds_iter = None
                continue

            except Exception as e:
                print(
                    f"Warning: Skipping problematic sample due to error: {e}")
                continue

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


class WdsGrainIterDataset(grain.IterDataset):

    def __init__(self, parent):
        super().__init__(parent)

    def __iter__(self) -> _WdsGrainIterator:
        # Return the wrapped, webdataset-powered iterator
        parent_iter = self._parent.__iter__()
        return _WdsGrainIterator(parent_iter)
