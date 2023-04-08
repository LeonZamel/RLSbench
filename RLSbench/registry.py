class Registry:
    def __init__(self, name):
        self.name = name
        self._registry = {}

    def register(self, name, object):
        if name in self._registry:
            raise ValueError(
                f"Registry {self.name} already contains an object registered with name {name}"
            )
        self._registry[name] = object

    def get(self, name):
        return self._registry[name]

    @property
    def available(self):
        return list(self._registry.keys())
