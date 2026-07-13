from dataclasses import dataclass

@dataclass
class IoCategory:
    want_return : bool
    want_output : bool
    has_input : bool

    @staticmethod
    def pure_input():
        return IoCategory( want_return = False, want_output = False, has_input = True )

    @staticmethod
    def for_return():
        return IoCategory( want_return = True, want_output = True, has_input = False )
