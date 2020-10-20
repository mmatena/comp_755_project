"""Some general interfaces around controllers."""


class Controller(object):
    @staticmethod
    def from_flat_arrays(array, in_size, out_size):
        raise NotImplementedError()

    @staticmethod
    def get_parameter_count(in_size, out_size):
        raise NotImplementedError()

    @staticmethod
    def deserialize(bytes_str):
        raise NotImplementedError()

    def sample_action(self, inputs):
        raise NotImplementedError()

    def in_size(self):
        raise NotImplementedError()

    def out_size(self):
        raise NotImplementedError()

    def serialize(self):
        raise NotImplementedError()
