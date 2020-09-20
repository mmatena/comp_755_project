"""General information about each environment."""
from enum import Enum, unique


class EnvironmentInfo(object):
    """General information about an environment."""

    def __init__(self, open_ai_name, folder_name, observation_shape, action_shape):
        """Create an EnvironmentInfo object.

        Args:
            open_ai_name: str, the name of the OpenAI gym environment
            folder_name: str, the name of the folder used when having
                code specific to this environment. For example, the
                "car_racing" in "rl755/models/car_racing".
            observation_shape: tuple[int], the shape of an individual
                observation
            action_shape: tuple[int], the shape of an individual action
        """
        self.open_ai_name = open_ai_name
        self.folder_name = folder_name
        self.observation_shape = observation_shape
        self.action_shape = action_shape


@unique
class Environments(Enum):
    """All supported environments."""

    # The car racing environment. Here, observations are a uint8 tensor with
    # shape [96,96,3] representing an image.
    #
    # Actions have shape [3] with each entry meaning:
    #     a[0]: [-1, 1], steering
    #     a[1]: [0, 1], gas
    #     a[2]: [0, 1], brakes
    # Values below/above the range are clipped to the min/max of the range, respectively.
    CAR_RACING = EnvironmentInfo(
        open_ai_name="CarRacing-v0",
        folder_name="car_racing",
        observation_shape=(96, 96, 3),
        action_shape=(3,),
    )

    @property
    def open_ai_name(self):
        return self.value.open_ai_name

    @property
    def folder_name(self):
        return self.value.folder_name
