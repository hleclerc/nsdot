from typing import TYPE_CHECKING, cast
from .DriverProxy import DriverProxy

from .FfiCode import FfiCode as FfiCode
from .FfiCode import FfiCodeCustom as FfiCodeCustom
from .FfiCode import FfiCodeParallel as FfiCodeParallel

# start with an unknown driver instance
driver = DriverProxy()

# give a concrete instance to typing
if TYPE_CHECKING:
    from .JaxDriver import JaxDriver
    driver = cast( JaxDriver, driver )
