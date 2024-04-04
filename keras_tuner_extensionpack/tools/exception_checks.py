"""Exception checks for the Keras Tuner Extension Pack."""

from __future__ import annotations


class NotInRangeError(Exception):
    """Exception raised for errors in the input value.

    Attributes:
        variable_name -- name of the variable
        variable_value -- input value which caused the error
        lower_bound -- lower bound of the range
        upper_bound -- upper bound of the range
        lower_bound_inclusive -- whether the lower bound is inclusive
        upper_bound_inclusive -- whether the upper bound is inclusive
    """

    def __init__(
        self,
        variable_name: str,
        variable_value: float,
        lower_bound: float = 0,
        upper_bound: float = 1,
        lower_bound_inclusive: bool = False,
        upper_bound_inclusive: bool = False,
    ) -> None:
        """
        Initialize a RangeError instance.

        Args:
            variable_name (str): The name of the variable that is out of range.
            variable_value (float): The value of the variable that is out of range.
            lower_bound (float, optional): The lower bound of the valid range. Defaults to 0.
            upper_bound (float, optional): The upper bound of the valid range. Defaults to 1.
            lower_bound_inclusive (bool, optional): Whether the lower bound is inclusive. Defaults to False.
            upper_bound_inclusive (bool, optional): Whether the upper bound is inclusive. Defaults to False.
        """

        self.variable_value = variable_value
        self.variable_name = variable_name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.lower_bound_inclusive = lower_bound_inclusive
        self.upper_bound_inclusive = upper_bound_inclusive
        lower_bound_symbol = "<=" if lower_bound_inclusive else "<"
        upper_bound_symbol = "<=" if upper_bound_inclusive else "<"
        full_message = (
            f"{self.__class__.__name__}: '{self.variable_name}={self.variable_value}' "
            f"is not in the range '{self.lower_bound} {lower_bound_symbol} "
            f"x {upper_bound_symbol} {self.upper_bound}'."
        )
        super().__init__(full_message)

    def __str__(self) -> str:
        """Return the string representation of the exception."""
        return str(self.args[0])


def not_in_range_check(
    variable_name: str,
    variable_value: float,
    *,
    lower_bound: float = 0,
    lower_bound_include: bool = False,
    upper_bound: float = 1,
    upper_bound_include: bool = False,
) -> None:
    """Checks if a variable is within a specified range.

    Args:
        variable_name (str): The name of the variable to check.
        variable_value (float): The value of the variable to check.
        lower_bound (float, optional): The lower bound of the range.
            Defaults to 0.
        lower_bound_include (bool, optional): Whether the range should
             include the lower bound. Defaults to False.
        upper_bound (float, optional): The upper bound of the range.
            Defaults to 1.
        upper_bound_include (bool, optional): Whether the range should
            include the upper bound. Defaults to False.

    Raises:
        NotInRangeError: If the variable value is not within the specified range.
    """
    lower_bound_check = (
        variable_value < lower_bound
        if lower_bound_include
        else variable_value <= lower_bound
    )
    upper_bound_check = (
        variable_value > upper_bound
        if upper_bound_include
        else variable_value >= upper_bound
    )

    if lower_bound_check or upper_bound_check:
        raise NotInRangeError(
            variable_name,
            variable_value,
            lower_bound,
            upper_bound,
            lower_bound_include,
            upper_bound_include,
        )
