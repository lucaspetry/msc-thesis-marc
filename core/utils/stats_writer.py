
class MetricsLogger:

    def __init__(self, keys=None, timestamp=True):
        self._stats = {}
        self._timestamp = timestamp
        self._keys = keys

        if self._keys is not None:
            for key in self._keys:
                self._stats[key] = []

        if self._timestamp:
            self._stats['timestamp'] = {}

    def log(self, **kwargs):
        if self._timestamp:
            self._stats['timestamp'].append(datetime.now())

        for key, value in kwargs.items():
            pass

    def save(self, file):
        self._df.to_csv(file, index=False)

    def load(self, file):
        if os.path.isfile(file):
            self._df = pd.read_csv(file)
        else:
            print("WARNING: File '" + file + "' not found!")

        return self
