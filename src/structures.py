"""
Data Structures.

This script defines the data structures needed for the algorithm of routine detection.

The module contains the following public classes

Subsequence: Basic data structure.
    Parameters:
        * instance: np.ndarray, the instance of the subsequence
        * date: datetime.date, the date of the subsequence
        * starting_point: int, the starting point of the subsequence

    Public methods:
        * get_instance: returns the instance of the subsequence
        * get_date: returns the date of the subsequence
        * get_starting_point: returns the starting point of the subsequence
        * to_collection: returns the subsequence as a dictionary
        * Magnitude: returns the magnitude of the subsequence
        * Distance: returns the distance between the subsequence and another subsequence or array

Sequence: Represents a sequence of subsequences
    Public Methods:
        * add_sequence: adds a subsequence to the sequence
        * get_by_starting_point: returns the subsequence with the specified starting point
        * set_by_starting_point: sets the subsequence with the specified starting point
        * get_starting_points: returns the starting points of the subsequences
        * get_dates: returns the dates of the subsequences
        * get_subsequences: returns the instances of the subsequences
        * to_collection: returns the sequence as a list of dictionaries

Cluster: Represents a cluster of subsequences
    Public Methods:
        * add_instance: adds a subsequence to the cluster
        * get_sequences: returns the sequences of the cluster
        * update_centroid: updates the centroid of the cluster
        * get_starting_points: returns the starting points of the subsequences
        * get_dates: returns the dates of the subsequences

Routines: Represents a collection of clusters
    Public Methods:
        * add_routine: adds a cluster to the collection
        * drop_indexes: drops the clusters with the specified indexes
        * get_routines: returns the clusters of the collection
        * to_collection: returns the collection as a list of dictionaries
"""

import numpy as np
import datetime
from typing import Union, Optional

import copy


class Subsequence:
    """
    Basic data structure.

    Parameters:
        * instance: np.ndarray, the instance of the subsequence
        * date: datetime.date, the date of the subsequence
        * starting_point: int, the starting point of the subsequence

    Public Methods:
        * get_instance: returns the instance of the subsequence
        * get_date: returns the date of the subsequence
        * get_starting_point: returns the starting point of the subsequence
        * to_collection: returns the subsequence as a dictionary
        * Magnitude: returns the magnitude of the subsequence
        * Distance: returns the distance between the subsequence and another subsequence or array

    Examples:
        >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        >>> subsequence.get_instance()
        np.array([1, 2, 3, 4])
        >>> subsequence.get_date()
        datetime.date(2021, 1, 1)
        >>> subsequence.get_starting_point()
        0
        >>> subsequence.to_collection()
        {'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}
        >>> subsequence.Magnitude()
        4
        >>> subsequence.Distance(np.array([1, 2, 3, 4]))
        0
    """

    def __init__(self, instance: np.ndarray, date: datetime.date, starting_point: int) -> None:
        """
        Parameters:
            * instance: `np.ndarray`, the instance of the subsequence
            * date: `datetime.date`, the date of the subsequence
            * starting_point: `int`, the starting point of the subsequence

        Raises:
            TypeError: if the parameters are not of the correct type

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
        """

        self.__checkType(instance, date, starting_point)
        self.__instance = instance
        self.__date = date
        self.__starting_point = starting_point

    @staticmethod
    def __checkType(instance: np.ndarray, date: datetime.date, starting_point: int) -> None:
        """
        Checks the type of the parameters

        Parameters:
            * instance: `np.ndarray`, the instance of the subsequence
            * date: `datetime.date`, the date of the subsequence
            * starting_point: `int`, the starting point of the subsequence

        Raises:
            TypeError: if the parameters are not of the correct type
        """

        # Check if the instance is an array
        if not isinstance(instance, np.ndarray):
            raise TypeError("Instances must be an arrays")

        # Check if the date is a datetime.date
        if not isinstance(date, datetime.date):
            raise TypeError("Date must be a timestamps")

        # Check if the starting point is an integer
        if not isinstance(starting_point, int):
            raise TypeError("starting_point must be a integer")

    def __repr__(self):
        """
        Returns the string representation of the subsequence

        Returns:
            str. The string representation of the subsequence

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> print(subsequence)
            Subsequence(
                instance=np.array([1, 2, 3, 4]),
                date=datetime.date(2021, 1, 1),
                starting_point=0
            )
        """

        return f"Subsequence(\n\t instance={self.__instance} \n\t date={self.__date} \n\t starting point = {self.__starting_point}\n)"

    def __str__(self):
        """
        Returns the string representation of the subsequence

        Returns:
            str. The string representation of the subsequence

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> print(subsequence)
            Subsequence(
                instance=np.array([1, 2, 3, 4]),
                date=datetime.date(2021, 1, 1),
                starting_point=0
            )
        """

        return f"Subsequence(\n\t instances={self.__instance} \n\t date={self.__date} \n\t starting point = {self.__starting_point}\n)"

    def __len__(self) -> int:
        """
        Returns the length of the subsequence

        Returns:
            `int`. The length of the subsequence

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> len(subsequence)
            4
        """
        return len(self.__instance)

    def __getitem__(self, index: int) -> int | float:
        """
        Get the item at the specified index in the subsequence

        Parameters:
            * index: `int`. The index of the item

        Returns:
            `float` | `int`. The item at the specified index in the subsequence

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> subsequence = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 1), 0)
            >>> subsequence[2]
            7
        """

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError("index must be an integer")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__instance):
            raise IndexError("index out of range")

        # If the item is a numpy integer or float, convert it to a Python integer or float and return it
        if isinstance(self.__instance[index], np.int32):
            return int(self.__instance[index])

        return float(self.__instance[index])

    def __eq__(self, other: 'Subsequence') -> bool:
        """
        Check if the subsequence is equal to another subsequence

        Parameters:
            other: `Subsequence`. The subsequence to compare

        Returns:
            `bool`. True if the subsequences are equal, False otherwise

        Raises:
            TypeError: if the parameter is not of the correct type

        Examples:
            >>> subsequence1 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence2 = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence1 == subsequence2
            True
        """

        # Check if the parameter is an instance of Subsequence
        if not isinstance(other, Subsequence):
            raise TypeError("other must be an instance of Subsequence")

        # Check if the instance, date, and starting point are equal
        if not np.array_equal(self.__instance, other.get_instance()):
            return False

        # Check if the date and starting point are equal
        if self.__date != other.get_date() or self.__starting_point != other.get_starting_point():
            return False

        return True

    def get_instance(self) -> np.ndarray:
        """
        Returns the instance of the subsequence

        Returns:
             `np.ndarray`. The instance of the `Subsequence`

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.get_instance()
            np.array([1, 2, 3, 4])
        """

        return self.__instance

    def get_date(self) -> datetime.date:
        """
        Returns the date of the subsequence

        Returns:
            `datetime.date`. The date of the `Subsequence`

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.get_date()
            datetime.date(2021, 1, 1)
        """

        return self.__date

    def get_starting_point(self) -> int:
        """
        Returns the starting point of the subsequence

        Returns:
             `int`. The starting point of the `Subsequence`

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.get_starting_point()
            0
        """

        return self.__starting_point

    def to_collection(self) -> dict:
        """
        Returns the subsequence as a dictionary

        Returns:
             `dict`. The subsequence as a dictionary

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.to_collection()
            {'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}
        """

        return {"instance": self.__instance, "date": self.__date, "starting_point": self.__starting_point}

    def Magnitude(self) -> float:
        """
        Returns the magnitude of the subsequence as the maximum value

        Returns:
             `float`. The magnitude of the subsequence

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.Magnitude()
            4.0
        """

        return np.max(self.__instance)

    def Distance(self, other: Union['Subsequence', np.ndarray]) -> float:
        """
        Returns the maximum absolute distance between the subsequence and another subsequence or array

        Parameters:
            * other: `Union[Subsequence, np.ndarray]`, the subsequence or array to compare

        Returns:
            `float`. The distance between the subsequence and another subsequence or array

        Raises:
            TypeError: if the parameter is not of the correct type

        Examples:
            >>> subsequence = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> subsequence.Distance(np.array([1, 2, 3, 4]))
            0.0

            >>> subsequence.Distance(Subsequence(np.array([1, 2, 3, 6]), datetime.date(2021, 1, 2), 2))
            2.0
        """

        # Check if the parameter is an instance of Subsequence or an array
        if isinstance(other, np.ndarray):
            return np.max(np.abs(self.__instance - other))

        if isinstance(other, Subsequence):
            return np.max(np.abs(self.__instance - other.get_instance()))

        # If the parameter is not an instance of Subsequence or an array, raise an error
        raise TypeError("other must be an instance of ndarray or a Subsequence")


class Sequence:
    """
    Represents a sequence of subsequences

    Parameters:
        * subsequence: `Union[Subsequence, None]`, the subsequence to add to the sequence

    Public Methods:
        * add_sequence: adds a `Subsequence` instance to the `Sequence`
        * get_by_starting_point: returns the subsequence with the specified starting point
        * set_by_starting_point: sets the subsequence with the specified starting point
        * get_starting_points: returns the starting points of the subsequences
        * get_dates: returns the dates of the subsequences
        * get_subsequences: returns the instances of the subsequences
        * to_collection: returns the sequence as a list of dictionaries

    Examples:
        >>> sequence = Sequence()
        >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
        >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
        >>> sequence.get_by_starting_point(0)
        Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)
        >>> sequence.get_starting_points()
        [0, 4]
        >>> sequence.get_dates()
        [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
        >>> sequence.get_subsequences()
        [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
        >>> sequence.to_collection()
        [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0},
         {'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}]
    """

    def __init__(self, subsequence: Optional[Subsequence] = None) -> None:
        """
        Parameters:
            * subsequence: Optional[Subsequence], the subsequence to add to the sequence

        Raises:
            TypeError: if the parameter is not of the correct type

        Examples:
            >>> sequence1 = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence2 = Sequence()
            >>> sequence2.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
        """

        # Check if the subsequence is a Subsequence instance
        if subsequence is not None:
            if not isinstance(subsequence, Subsequence):
                raise TypeError("subsequence has to be an instance of Subsequence")

            # Make a deep copy of the subsequence
            new_subsequence = copy.deepcopy(subsequence)
            self.__list_sequences: list[Subsequence] = [new_subsequence]

        # If the subsequence is None, initialize an empty list
        else:
            self.__list_sequences: list[Subsequence] = []

    def __repr__(self):
        """
        Returns the string representation of the sequence

        Returns:
            str. The string representation of the sequence

        Examples:
            >>> sequence = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> print(sequence)
            Sequence(
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    )
                ]
            )
        """

        out_string = "Sequence(\n\tlist_sequences=[\n"
        for seq in self.__list_sequences:
            out_string += f" {seq},\n"

        out_string = out_string[:-2] + out_string[-1] + "]"
        return out_string

    def __str__(self):
        """
        Returns the string representation of the sequence

        Returns:
            `str`. The string representation of the sequence

        Examples:
            >>> sequence = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> print(sequence)
            Sequence(
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    )
                ]
            )
        """
        out_string = "Sequence(\n\tlist_sequences=[\n"
        for seq in self.__list_sequences:
            out_string += f" {seq},\n"

        out_string = out_string[:-2] + out_string[-1] + "]"
        return out_string

    def __len__(self) -> int:
        """
        Returns the number of `Subsequence` instances in the `Sequence`

        Returns:
            `int`. The number of subsequences in the sequence

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> len(sequence)
            2
        """

        return len(self.__list_sequences)

    def __getitem__(self, index: int) -> 'Subsequence':
        """
        Get the subsequence at the specified index in the sequence

        Parameters:
            * index: `int`. The index of the subsequence

        Returns:
            `Subsequence`. The subsequence at the specified index in the sequence

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence[0]
            Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)
        """
        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError("index must be an integer")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__list_sequences):
            raise IndexError("index out of range")

        return self.__list_sequences[index]

    def __setitem__(self, index: int, value: 'Subsequence') -> None:
        """
        Set the value of the subsequence at the specified index in the sequence

        Parameters:
            * index: int. The index of the subsequence
            * value: Subsequence. The new subsequence

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence[0] = Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4)
            >>> sequence[0]
            Subsequence(instance=np.array([5, 6, 7, 8]), date=datetime.date(2021, 1, 2), starting_point=4)
        """

        # Check if the new_sequence is a Subsequence instance
        if not isinstance(value, Subsequence):
            raise TypeError("new_sequence must be an instance of Subsequence")

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError("index must be an integer")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__list_sequences):
            raise IndexError("index out of range")

        self.__list_sequences[index] = value

    def __iter__(self):
        """
        Returns an iterator for each subsequence in the sequence

        Returns:
            iter. An iterator for each subsequence in the sequence

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> for subsequence in sequence:
            ...     print(subsequence)
            Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)
            Subsequence(instance=np.array([5, 6, 7, 8]), date=datetime.date(2021, 1, 2), starting_point=4)
        """

        return iter(self.__list_sequences)

    def __contains__(self, item: 'Subsequence') -> bool:
        """
        Check if the subsequence exists in the sequence

        Parameters:
            * item: `Subsequence`. The subsequence to check

        Returns:
            `bool`. `True` if the subsequence exists, `False` otherwise

        Raises:
            TypeError: if the parameter is not an instance of Subsequence

        Examples:
            >>> sequence = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0) in sequence
            True

        """
        # Check if the new_sequence is a Subsequence instance
        if not isinstance(item, Subsequence):
            raise TypeError("new_sequence must be an instance of Subsequence")

        return item in self.__list_sequences

    def __delitem__(self, index: int) -> None:
        """
        Deletes the subsequence at the specified index in the sequence

        Parameters:
            * index: `int`. The index of the subsequence to delete

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range
        """
        if not isinstance(index, int):
            raise TypeError("index must be an integer")

        if not 0 <= index < len(self.__list_sequences):
            raise IndexError("index out of range")

        del self.__list_sequences[index]

    def __add__(self, other: 'Sequence') -> 'Sequence':
        """
        Concatenates two sequences together with the operator +

        Parameters:
            * other: `Sequence`. The sequence to concatenate

        Returns:
            `Sequence`. The concatenated sequence

        Raises:
            TypeError: if the parameter is not an instance of `Sequence`

        Examples:
            >>> sequence1 = Sequence()
            >>> sequence1.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence2 = Sequence()
            >>> sequence2.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> new_sequence = sequence1 + sequence2
            >>> print(new_sequence)
            Sequence(
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    ),
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=1
                    )
                ]
            )
        """
        if not isinstance(other, Sequence):
            raise TypeError("other must be an instance of Sequence")

        new_sequence = Sequence()
        new_sequence.__list_sequences = self.__list_sequences + other.__list_sequences
        return new_sequence

    def __eq__(self, other: 'Sequence') -> bool:
        """
        Check if the sequence is equal to another sequence

        Parameters:
            * other: `Sequence`. The sequence to compare

        Returns:
            `bool`. True if the sequences are equal, False otherwise

        Raises:
            TypeError: if the parameter is not of the correct type

        Examples:
            >>> sequence1 = Sequence()
            >>> sequence1.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence2 = Sequence()
            >>> sequence2.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence1 == sequence2
            True
        """

        # Check if the parameter is an instance of Sequence
        if not isinstance(other, Sequence):
            raise TypeError("other must be an instance of Sequence")

        # Check if the subsequences are equal
        return np.array_equal(self.get_subsequences(), other.get_subsequences())

    def _alreadyExists(self, subsequence: 'Subsequence') -> bool:
        """
        Check if the subsequence already exists in the sequence

        Parameters:
            * subsequence: `Subsequence`. The subsequence to check

        Returns:
            `bool`. True if the `subsequence` already exists, `False` otherwise

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence._alreadyExists(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            True
        """

        self_collection = self.to_collection()
        new_self_collection = []

        # Is necessary to convert the arrays to list for checking properly if the new sequence exists
        for idx, dictionary in enumerate(self_collection):
            dictionary["instance"] = dictionary["instance"].tolist()
            new_self_collection.append(dictionary)

        # convert to collection and transform from array to list
        collection = subsequence.to_collection()
        collection = {"instance": collection["instance"].tolist()}

        return collection in new_self_collection

    def add_sequence(self, new: 'Subsequence') -> None:
        """
        Adds a subsequence to the sequence

        Parameters:
            * new: `Subsequence`. The subsequence to add

        Raises:
            TypeError: if the parameter is not of the correct type
            RuntimeError: if the subsequence already exists

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> print(sequence)
            Sequence(
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    ),
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=4
                    )
                ]
            )
        """
        # Check if the new sequence is a Subsequence instance
        if not isinstance(new, Subsequence):
            raise TypeError("new has to be an instance of Subsequence")

        # Check if the new sequence already exists
        if self._alreadyExists(new):
            raise RuntimeError("new sequence already exists ")

        self.__list_sequences.append(new)

    def get_by_starting_point(self, starting_point: int) -> Optional['Subsequence']:
        """
        Returns the subsequence with the specified starting point

        Parameters:
            * starting_point: `int`. The starting point of the subsequence

        Returns:
            Optional[Subsequence]. The subsequence with the specified starting point if it exists. Otherwise, None

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> sequence.get_by_starting_point(0)
            Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)

            >>> sequence.get_by_starting_point(2)
            None
        """

        for subseq in self.__list_sequences:
            if subseq.get_starting_point() == starting_point:
                return subseq

        return None

    def set_by_starting_point(self, starting_point: int, new_sequence: 'Subsequence') -> None:
        """
        Sets the subsequence with the specified starting point

        Parameters:
            * starting_point: int. The starting point of the subsequence
            * new_sequence: Subsequence. The new subsequence

        Raises:
            TypeError: if the parameter is not of the correct type
            ValueError: if the starting point does not exist

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> sequence.set_by_starting_point(0, Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 0))
            >>> sequence.get_by_starting_point(0)
            Subsequence(instance=np.array([9, 10, 11, 12]), date=datetime.date(2021, 1, 3), starting_point=0)
        """

        # Check if the new_sequence is a Subsequence instance
        if not isinstance(new_sequence, Subsequence):
            raise TypeError("new_sequence must be an instance of Subsequence")

        # Iterate through the list to find the subsequence with the matching starting point
        for i, subseq in enumerate(self.__list_sequences):
            if subseq.get_starting_point() == starting_point:
                # Replace the found subsequence with the new one
                self.__list_sequences[i] = new_sequence
                return

        # If not found, raise an error indicating the starting point does not exist
        raise ValueError("The starting point doesn't exist")

    def get_starting_points(self) -> list[int]:
        """
        Returns the starting points of the subsequences

        Returns:
             `list[int]`. The starting points of the subsequences

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> sequence.get_starting_points()
            [0, 4]
        """
        return [subseq.get_starting_point() for subseq in self.__list_sequences]

    def get_dates(self) -> list[datetime.date]:
        """
        Returns the dates of the subsequences

        Returns:
             `list[datetime.date]`. The dates of the subsequences

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> sequence.get_dates()
            [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
        """

        return [subseq.get_date() for subseq in self.__list_sequences]

    def get_subsequences(self) -> list[np.ndarray]:
        """
        Returns the instances of the subsequences

        Returns:
             `list[np.ndarray]`. The instances of the subsequences

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> sequence.get_subsequences()
            [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
        """
        return [subseq.get_instance() for subseq in self.__list_sequences]

    def to_collection(self) -> list[dict]:
        """
        Returns the sequence as a list of dictionaries

        Returns:
             `list[dict]`. The sequence as a list of dictionaries
        """
        collection = []
        for subseq in self.__list_sequences:
            collection.append({
                'instance': subseq.get_instance(),
                'date': subseq.get_date(),
                'starting_point': subseq.get_starting_point()
            })

        return collection


class Cluster:
    """
    Represents a cluster of subsequences from a sequence and a centroid.

    Parameters:
        * centroid: `np.ndarray`, the centroid of the cluster
        * instances: `Sequence`, the sequence of subsequences

    Public Methods:
        * add_instance: adds a subsequence to the cluster
        * get_sequences: returns the sequences of the cluster
        * update_centroid: updates the centroid of the cluster
        * get_starting_points: returns the starting points of the subsequences
        * get_dates: returns the dates of the subsequences

    Examples:
        >>> sequence = Sequence()
        >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
        >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
        >>> cluster = Cluster(np.array([1, 1, 1, 1]), sequence)
        >>> cluster.get_sequences().to_collection()
        [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0},
         {'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}]
        >>> cluster.get_starting_points()
        [0, 4]
        >>> cluster.get_dates()
        [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
        >>> cluster.centroid
        np.array([1, 1, 1, 1])
        >>> cluster.centroid = np.array([1, 2, 3, 4])
        >>> cluster.centroid
        np.array([1, 2, 3, 4])
        >>> cluster.update_centroid()
        >>> cluster.centroid
        np.array([3, 4, 5, 6])
    """

    def __init__(self, centroid: np.ndarray, instances: 'Sequence') -> None:
        """
        Parameters:
            * centroid: `np.ndarray`, the centroid of the cluster
            * instances: `Sequence`, the sequence of subsequences

        Raises:
            TypeError: if the centroid is not an instance of `np.ndarray` or the instances are not an instance of `Sequence`

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
        """
        if not isinstance(centroid, np.ndarray):
            raise TypeError("centroid must be an instance of np.ndarray")

        if not isinstance(instances, Sequence):
            raise TypeError("instances must be an instance of Sequence")

        # Make a deep copy of the instances to avoid modifying the original sequence
        new_instances = copy.deepcopy(instances)

        self.__centroid: np.ndarray = centroid
        self.__instances: Sequence = new_instances

    def __str__(self):
        """
        Returns the string representation of the cluster

        Returns:
            `str`. The string representation of the cluster

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> print(cluster)
            Cluster(
                -Centroid: np.array([3, 4, 5, 6])
                -Instances: [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
                -Dates: [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
                -Starting Points: [0, 4]
            )
        """

        out_string = f"Cluster(\n\t -Centroid: {self.__centroid} \n"
        out_string += f"\t -Instances: {[instance for instance in self.__instances.get_subsequences()]}\n"
        out_string += f"\t -Dates: {[date for date in self.__instances.get_dates()]}\n"
        out_string += f"\t -Starting Points: {[sp for sp in self.__instances.get_starting_points()]}\n)"
        return out_string

    def __repr__(self):
        """
        Returns the string representation of the cluster

        Returns:
            `str`. The string representation of the cluster

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> print(cluster)
            Cluster(
                -Centroid: np.array([3, 4, 5, 6])
                -Instances: [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
                -Dates: [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
                -Starting Points: [0, 4]
            )
        """

        out_string = f"Cluster(\n\t -Centroid: {self.__centroid} \n"
        out_string += f"\t -Instances: {[instance for instance in self.__instances.get_subsequences()]}\n"
        out_string += f"\t -Dates: {[date for date in self.__instances.get_dates()]}\n"
        out_string += f"\t -Starting Points: {[sp for sp in self.__instances.get_starting_points()]}\n)"
        return out_string

    def __len__(self) -> int:
        """
        Returns the number of instances in the cluster

        Returns:
            `int`. The number of instances in the cluster

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> len(cluster)
            2
        """

        return len(self.__instances)

    def __getitem__(self, index: int) -> 'Subsequence':
        """
        Get the subsequence at the specified index in the cluster

        Parameters:
            * index: `int`. The index of the subsequence

        Returns:
            `Subsequence`. The subsequence at the specified index in the cluster

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> cluster[0]
            Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)
        """

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError("index must be an integer")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__instances):
            raise IndexError("index out of range")

        return self.__instances[index]

    def __setitem__(self, index: int, value: 'Subsequence') -> None:
        """
        Set the value of the subsequence at the specified index in the cluster

        Parameters:
            * index: `int`. The index of the subsequence
            * value: `Subsequence`. The new subsequence

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> cluster[0] = Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 0)
            >>> cluster[0]
            Subsequence(instance=np.array([9, 10, 11, 12]), date=datetime.date(2021, 1, 3), starting_point=0)
        """

        # Check if the new_sequence is a Subsequence instance
        if not isinstance(value, Subsequence):
            raise TypeError("new_sequence must be an instance of Subsequence")

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError("index must be an integer")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__instances):
            raise IndexError("index out of range")

        self.__instances[index] = value

    def __iter__(self) -> iter:
        """
        Returns an iterator for each subsequence in the cluster's instances

        Returns:
            `iter`. An iterator for each subsequence in the cluster's instances

        Examples:
            >>> sequence = Sequence()
            >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> for subsequence in cluster:
            ...     print(subsequence)
            Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)
            Subsequence(instance=np.array([5, 6, 7, 8]), date=datetime.date(2021, 1, 2), starting_point=4)
        """

        return iter(self.__instances)

    def __contains__(self, item: 'Subsequence') -> bool:
        """
        Check if the subsequence exists in the cluster's instances

        Parameters:
            * item: `Subsequence`. The subsequence to check

        Returns:
            `bool`. `True` if the subsequence exists, `False` otherwise

        Raises:
            TypeError: if the parameter is not an instance of `Subsequence`

        Examples:
            >>> sequence = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> item = Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)
            >>> item in cluster
            True
        """

        # Check if the new_sequence is a Subsequence instance
        if not isinstance(item, Subsequence):
            raise TypeError("new_sequence must be an instance of Subsequence")

        return item in self.__instances

    def __delitem__(self, index: int) -> None:
        """
        Deletes the subsequence at the specified index in the cluster's instances

        Parameters:
            * index: `int`. The index of the subsequence to delete

        Raises:
            TypeError: if the parameter is not of the correct type
            IndexError: if the index is out of range

        Examples:
            >>> sequence = Sequence(subsequence=Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> del cluster[0]
            >>> print(cluster)
            Cluster(
                -Centroid: np.array([3, 4, 5, 6])
                -Instances: [np.array([5, 6, 7, 8])]
                -Dates: [datetime.date(2021, 1, 2)]
                -Starting Points: [4]
            )
        """

        if not isinstance(index, int):
            raise TypeError("index must be an integer")

        if not 0 <= index < len(self.__instances):
            raise IndexError("index out of range")

        del self.__instances[index]

    def __add__(self, other: 'Cluster') -> 'Cluster':
        """
        Concatenates two clusters together with the operator + and updates the centroid

        Parameters:
            * other: `Cluster`. The cluster to concatenate

        Returns:
            `Cluster`. The concatenated cluster

        Raises:
            TypeError: if the parameter is not an instance of `Cluster`
            ValueError: if the clusters do not have the same length of instances in each `Subsequence`

        Examples:
            >>> sequence1 = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence2 = Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), sequence1)
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), sequence2)
            >>> new_cluster = cluster1 + cluster2
            >>> print(new_cluster)
            Cluster(
                -Centroid: np.array([5, 6, 7, 8])
                -Instances: [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
                -Dates: [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
                -Starting Points: [0, 1]
            )
        """

        if not isinstance(other, Cluster):
            raise TypeError("other must be an instance of Cluster")

        # Check if the lengths of the subsequences from the instances of each cluster match
        if len(self.__instances[0]) != len(other.get_sequences()[0]):
            raise ValueError("clusters do not have the same length of instances in each Subsequence")

        new_instances = self.__instances + other.get_sequences()
        new_centroid = np.mean(new_instances.get_subsequences(), axis=0)
        return Cluster(centroid=new_centroid, instances=new_instances)

    def __eq__(self, other: Union['Cluster', None]) -> bool:
        """
        Check if the cluster is equal to another cluster with the operator ==

        Parameters:
            * other: `Cluster`. The cluster to check

        Returns:
            `bool`. `True` if the clusters are equal, `False` otherwise

        Raises:
            TypeError: if the parameter is not an instance of `Cluster`

        Examples:
            >>> sequence1 = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence2 = Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), sequence1)
            >>> cluster2 = Cluster(np.array([3, 4, 5, 6]), sequence2)
            >>> cluster1 == cluster2
            False

            >>> sequence3 = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence4 = Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 1))
            >>> cluster3 = Cluster(np.array([3, 4, 5, 6]), sequence3)
            >>> cluster4 = Cluster(np.array([3, 4, 5, 6]), sequence4)
            >>> cluster3 == cluster4
            True
        """

        # Check if the other is a Cluster instance
        if not isinstance(other, Cluster):
            if other is None:
                return False

            raise TypeError("other must be an instance of Cluster")

        # Check if the centroid and the instances are equal
        if not np.array_equal(self.__centroid, other.centroid):
            return False

        # Check if the number of instances is equal
        if len(self.__instances) != len(other.get_sequences()):
            return False

        # Check if the instances are equal
        if not np.array_equal(self.__instances.get_subsequences(), other.get_sequences().get_subsequences()):
            return False

        # Check if the dates are equal
        if not np.array_equal(self.__instances.get_dates(), other.get_sequences().get_dates()):
            return False

        # Check if the starting points are equal
        if not np.array_equal(self.__instances.get_starting_points(), other.get_sequences().get_starting_points()):
            return False

        return True

    def add_instance(self, new_instance: 'Subsequence') -> None:
        """
        Adds a subsequence to the instances of the cluster

        Parameters:
            * new_instance: `Subsequence`. The subsequence to add

        Raises:
            TypeError: if the parameter is not of the correct type
            ValueError: if the subsequence is already an instance of the cluster

        Examples:
            >>> sequence = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> cluster.add_instance(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 0))
            >>> print(cluster)
            Cluster(
                -Centroid: np.array([3, 4, 5, 6])
                -Instances: [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]), np.array([9, 10, 11, 12])]
                -Dates: [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2), datetime.date(2021, 1, 3)]
                -Starting Points: [0, 4, 0]
            )
        """

        # Check if the new_sequence is a Subsequence instance
        if not isinstance(new_instance, Subsequence):
            raise TypeError("new sequence must be an instance of Subsequence")

        # Check if the new sequence is already an instance of the cluster
        if self.__instances._alreadyExists(new_instance):
            raise ValueError("new sequence is already an instance of the cluster")

        self.__instances.add_sequence(new_instance)

    def get_sequences(self) -> 'Sequence':
        """
        Returns the sequence of the cluster

        Returns:
             `Sequence`. The sequence of the cluster

        Examples:
            >>> sequence = Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
            >>> cluster.get_sequences()
            Sequence(
                list_sequences=[
                    Subsequence(
                        instance=np.array([1, 2, 3, 4]),
                        date=datetime.date(2021, 1, 1),
                        starting_point=0
                    ),
                    Subsequence(
                        instance=np.array([5, 6, 7, 8]),
                        date=datetime.date(2021, 1, 2),
                        starting_point=4
                    )
                ]
            )
        """

        return self.__instances

    def update_centroid(self) -> None:
        """
        Updates the centroid of the cluster with the mean of the instances
        """
        self.__centroid = np.mean(self.__instances.get_subsequences(), axis=0)

    @property
    def centroid(self) -> np.ndarray:
        """
        Returns the centroid of the cluster
        :return: np.ndarray. The centroid of the cluster
        """
        return self.__centroid

    @centroid.setter
    def centroid(self, subsequence: np.ndarray | Subsequence) -> None:
        """
        Sets the value of the centroid of the cluster from a subsequence

        Parameters:
            * subsequence: `Union[Subsequence|np.ndarray]`. The subsequence to set as the centroid

        Raises:
            TypeError: if the parameter is not a `Subsequence` or a numpy array
        """
        if isinstance(subsequence, Subsequence):
            self.__centroid = subsequence.get_instance()

        if isinstance(subsequence, np.ndarray):
            self.__centroid = subsequence

        if not isinstance(subsequence, Subsequence) and not isinstance(subsequence, np.ndarray):
            raise TypeError(f"subsequence must be an instance of Subsequence or a numpy array")

    def get_starting_points(self) -> list[int]:
        """
        Returns the starting points of the subsequences

        Returns:
             `list[int]`. The starting points of the subsequences
        """

        return self.__instances.get_starting_points()

    def get_dates(self) -> list[datetime.date]:
        """
        Returns the dates of the subsequences

        Returns:
             `list[datetime.date]`. The dates of the subsequences
        """

        return self.__instances.get_dates()

    def cumulative_magnitude(self) -> float | int:
        """
        Returns the magnitude's sum of the subsequences that belongs to the instances within the cluster

        Returns:
             `float`. The magnitude's sum of the subsequences
        """

        return sum([subsequence.Magnitude() for subsequence in self.__instances])


class Routines:
    """
    Represents a collection of clusters, each of them representing a routine.

    Parameters:
        * cluster: Union[Cluster, None], the cluster to add to the collection

    Public Methods:
        * add_routine: adds a cluster to the collection
        * drop_indexes: drops the clusters with the specified indexes
        * get_routines: returns the clusters of the collection
        * get_centroids: returns the centroids of the clusters
        * to_collection: returns the collection as a list of dictionaries

    Examples:
        >>> sequence = Sequence()
        >>> sequence.add_sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
        >>> sequence.add_sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
        >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
        >>> routines = Routines(cluster)
        >>> routines.get_routines()
        [Cluster(centroid=np.array([3, 4, 5, 6]), instances=Sequence(list_sequences=[Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0), Subsequence(instance=np.array([5, 6, 7, 8]), date=datetime.date(2021, 1, 2), starting_point=4)]))]
        >>> routines.add_routine(cluster)
        >>> routines.get_routines()
        [Cluster(centroid=np.array([3, 4, 5, 6]), instances=Sequence(list_sequences=[Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0), Subsequence(instance=np.array([5, 6, 7, 8]), date=datetime.date(2021, 1, 2), starting_point=4)])), Cluster(centroid=np.array([3, 4, 5, 6]), instances=Sequence(list_sequences=[Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0), Subsequence(instance=np.array([5, 6, 7, 8]), date=datetime.date(2021, 1, 2), starting_point=4)]))]
        >>> routines.drop_indexes([0])
        [Cluster(centroid=np.array([3, 4, 5, 6]), instances=Sequence(list_sequences=[Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0), Subsequence(instance=np.array([5, 6, 7, 8
    """

    def __init__(self, cluster: Optional[Cluster] = None) -> None:
        """
        Parameters:
            * cluster: `Optional[Cluster]`, the cluster to add to the routines. Default is None

        Raises:
             TypeError: if the parameter is not an instance of cluster

        Examples:
             >>> routines = Routines()
             >>> print(routines)
             Routines(
                list_routines = [[]]
             )

             >>> sequence = Sequence(subsequence=Subsequence(np.array([1,2,3], datetime.date(2024, 1, 1), 1)))
             >>> routines = Routines(Cluster(centroid=np.array([1,2,3], instances=sequence)))
             >>> print(routines)
             Routines(
                list_routines = [
                    Cluster(
                        - centroid = [1,2,3],
                        - instances = [[1,2,3]]
                        - starting_points = [1]
                        - dates = [datetime.date(2024, 1, 1)]
                    )
                ]
             )
        """

        if cluster is not None:
            if not isinstance(cluster, Cluster):
                raise TypeError("cluster has to be an instance of Cluster")

            self.__routines: list[Cluster] = [cluster]
        else:
            self.__routines: list[Cluster] = []

    def __repr__(self):
        """
        Returns the string representation of the routines

        Returns:
            `str`. The string representation of the routines

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> print(routines)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5, 6],
                        - instances = [[1, 2, 3, 4]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    ),
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    )
                ]
            )
        """

        out_string = "Routines(\n\tlist_routines=[\n"
        for routine in self.__routines:
            out_string += f" {routine},\n"

        out_string = out_string[:-2] + out_string[-1] + "])"
        return out_string

    def __str__(self):
        """
        Returns the string representation of the routines

        Returns:
            `str`. The string representation of the routines

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> print(routines)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5, 6],
                        - instances = [[1, 2, 3, 4]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    ),
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    )
                ]
            )
        """

        out_string = "Routines(\n\tlist_routines=[\n"
        for routine in self.__routines:
            out_string += f" {routine},\n"

        out_string = out_string[:-2] + out_string[-1] + "])"
        return out_string

    def __len__(self) -> int:
        """
        Returns the number of clusters in the `Routines`

        Returns:
            `int`. The number of clusters in the `Routines`

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> len(routines)
            2
        """

        return len(self.__routines)

    def __getitem__(self, index: int) -> 'Cluster':
        """
        Get the cluster at the specified index in the collection

        Parameters:
            * index: `int`. The index of the cluster

        Returns:
            `Cluster`. The cluster at the specified index in the collection

        Raises:
            TypeError: if the index is not an integer
            IndexError: if the index is out of range of the routines

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> print(routines[0])
            Cluster(
                - centroid = [3, 4, 5, 6],
                - instances = [[1, 2, 3, 4]]
                - starting_points = [0]
                - dates = [datetime.date(2021, 1, 1)]
            )
        """
        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError("index must be an integer")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__routines):
            raise IndexError("index out of range")

        return self.__routines[index]

    def __setitem__(self, index: int, value: 'Cluster') -> None:
        """
        Set the value of the cluster at the specified index in the collection

        Parameters:
            * index: `int`. The index of the cluster
            * value: `Cluster`. The new cluster

        Raises:
            TypeError: if the index is not an integer or the value is not an instance of Cluster
            IndexError: if the index is out of range of the routines

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> routines[0] = Cluster(np.array([11, 12, 13, 14]), Sequence(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 4))
            >>> print(routines[0])
            Cluster(
                - centroid = [11, 12, 13, 14],
                - instances = [[9, 10, 11, 12]]
                - starting_points = [4]
                - dates = [datetime.date(2021, 1, 3)]
            )
        """
        # Check if the value is a Cluster instance
        if not isinstance(value, Cluster):
            raise TypeError("value has to be an instance of Cluster")

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError("index has to be an integer")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__routines):
            raise IndexError("index out of range")

        self.__routines[index] = value

    def __iter__(self) -> iter:
        """
        Returns an iterator for each cluster in the `Routines`

        Returns:
            iter. An iterator for each cluster in the `Routines`

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> for cluster in routines:
            ...     print(cluster)
            Cluster(
                - centroid = [3, 4, 5, 6],
                - instances = [[1, 2, 3, 4]]
                - starting_points = [0]
                - dates = [datetime.date(2021, 1, 1)]
            )
            Cluster(
                - centroid = [7, 8, 9, 10],
                - instances = [[5, 6, 7, 8]]
                - starting_points = [4]
                - dates = [datetime.date(2021, 1, 2)]
            )
        """

        return iter(self.__routines)

    def __contains__(self, item: 'Cluster') -> bool:
        """
        Check if the cluster exists in the collection

        Parameters:
            * item: `Cluster`. The cluster to check

        Returns:
            `bool`. `True` if the cluster exists, `False` otherwise

        Raises:
            TypeError: if the parameter is not an instance of Cluster
        """

        # Check if the item is a Cluster instance
        if not isinstance(item, Cluster):
            raise TypeError("item has to be an instance of Cluster")

        return item in self.__routines

    def __delitem__(self, index: int) -> None:
        """
        Deletes the cluster at the specified index in the collection

        Parameters:
            * index: `int`. The index of the cluster to delete

        Returns:
            `Cluster`. The cluster at the specified index in the collection

        Raises:
            TypeError: if the index is not an integer
            IndexError: if the index is out of range of the routines
        """

        # Check if the index is an integer
        if not isinstance(index, int):
            raise TypeError("index has to be an integer")

        # Check if the index is within the range of the list
        if not 0 <= index < len(self.__routines):
            raise IndexError("index out of range")

        del self.__routines[index]

    def __add__(self, other: 'Routines') -> 'Routines':
        """
        Concatenates two routines together with the operator + and returns a new collection

        Parameters:
            * other: `Routines`. The collection to concatenate

        Returns:
            `Routines`. The concatenated `Routines`

        Raises:
            TypeError: if the parameter is not an instance of Routines

        Examples:
            >>> routines1 = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines1.add_routine(cluster1)
            >>> routines1.add_routine(cluster2)
            >>> routines2 = Routines()
            >>> cluster3 = Cluster(np.array([11, 12, 13, 14]), Sequence(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 4))
            >>> routines2.add_routine(cluster3)
            >>> new_routines = routines1 + routines2
            >>> print(new_routines)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5, 6],
                        - instances = [[1, 2, 3, 4]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    ),
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    ),
                    Cluster(
                        - centroid = [11, 12, 13, 14],
                        - instances = [[9, 10, 11, 12]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 3)]
                    )
                ]
            )
        """

        # Check if the other is a Routines instance
        if not isinstance(other, Routines):
            raise TypeError("other has to be an instance of Routines")

        new_routines = Routines()
        new_routines.__routines = self.__routines + other.__routines
        return new_routines

    def __eq__(self, other: 'Routines') -> bool:
        """
        Check if the self routine is equal to another routine with the operator ==

        Parameters:
            * other: `Routines`. The routine to check

        Returns:
            `bool`. `True` if the routines are equal, `False` otherwise

        Raises:
            TypeError: if the parameter is not an instance of `Routines`

        Examples:
            >>> routines1 = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines1.add_routine(cluster1)
            >>> routines1.add_routine(cluster2)
            >>> routines2 = Routines()
            >>> cluster3 = Cluster(np.array([11, 12, 13, 14]), Sequence(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 4))
            >>> routines2.add_routine(cluster3)
            >>> routines1 == routines2
            False

            >>> routines3 = Routines()
            >>> cluster4 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster5 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines3.add_routine(cluster4)
            >>> routines3.add_routine(cluster5)
            >>> routines1 == routines3
            True
        """

        # Check if the other is a Routines instance
        if not isinstance(other, Routines):
            raise TypeError("other has to be an instance of Routines")

        # Check if the number of clusters is equal
        if len(self.__routines) != len(other.__routines):
            return False

        # Check if the clusters are equal
        for idx, routine in enumerate(self.__routines):
            if routine != other.__routines[idx]:
                return False

        return True

    def add_routine(self, new_routine: 'Cluster') -> None:
        """
        Adds a cluster to the collection

        Parameters:
            new_routine: `Cluster`. The cluster to add

        Raises:
             TypeError: if the parameter is not of the correct type

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> print(routines)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid = [3, 4, 5, 6],
                        - instances = [[1, 2, 3, 4]]
                        - starting_points = [0]
                        - dates = [datetime.date(2021, 1, 1)]
                    ),
                    Cluster(
                        - centroid = [7, 8, 9, 10],
                        - instances = [[5, 6, 7, 8]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 2)]
                    )
                ]
            )
        """

        # Check if the new_routine is a Cluster instance
        if not isinstance(new_routine, Cluster):
            raise TypeError("new_routine has to be an instance of Cluster")

        self.__routines.append(new_routine)

    def drop_indexes(self, to_drop: list[int]) -> 'Routines':
        """
        Drops the clusters with the specified indexes

        Parameters:
            to_drop: `list[int]`. The indexes of the clusters to drop

        Returns:
             Routines. The collection without the dropped clusters


        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0)))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 2)))
            >>> cluster3 = Cluster(np.array([11, 12, 13, 14]), Sequence(Subsequence(np.array([9, 10, 11, 12]), datetime.date(2021, 1, 3), 4)))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> routines.add_routine(cluster3)
            >>> filtered_routines = routines.drop_indexes([0, 2])
            >>> print(filtered_routines)
            Routines(
                list_routines=[
                    Cluster(
                        - centroid = [11, 12, 13, 14],
                        - instances = [[9, 10, 11, 12]]
                        - starting_points = [4]
                        - dates = [datetime.date(2021, 1, 3)]
                    )
                ]
            )

        Notes:
            This method does not modify the original `Routine`, it returns a new one without the dropped clusters
        """

        new_routines = Routines()
        for idx, cluster in enumerate(self.__routines):
            if idx not in to_drop:
                new_routines.add_routine(cluster)
        return new_routines

    def get_routines(self) -> list[Cluster]:
        """
        Returns the clusters of the `Routines`

        Returns
            `list[Cluster]`. Returns all the clusters of the routines as a list of clusters

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> routines.get_routines()
            [Cluster(centroid=np.array([3, 4, 5, 6]), instances=Sequence(list_sequences=[Subsequence(instance=np.array([1, 2, 3, 4]), date=datetime.date(2021, 1, 1), starting_point=0)]), Cluster(centroid=np.array([7, 8, 9, 10]), instances=Sequence(list_sequences=[Subsequence(instance=np.array([5, 6, 7, 8]), date=datetime.date(2021, 1, 2), starting_point=4)])]
        """

        return self.__routines

    def get_centroids(self) -> list[np.ndarray]:
        """
        Returns the centroids of the clusters

        Returns:
             `list[np.ndarray]`. The centroids of the clusters

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> routines.get_centroids()
            [np.array([3, 4, 5, 6]), np.array([7, 8, 9, 10])]
        """

        return [cluster.centroid for cluster in self.__routines]

    def to_collection(self) -> list[dict]:
        """
        Returns the collection as a list of dictionaries

        Returns:
             `list[dict]`. The routines as a list of dictionaries

        Examples:
            >>> routines = Routines()
            >>> cluster1 = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> cluster2 = Cluster(np.array([7, 8, 9, 10]), Sequence(Subsequence(np.array([5, 6, 7, 8]), datetime.date(2021, 1, 2), 4))
            >>> routines.add_routine(cluster1)
            >>> routines.add_routine(cluster2)
            >>> routines.to_collection()
            [{ 'centroid': np.array([3, 4, 5, 6]),
              'instances': [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0}]},
              {'centroid': np.array([7, 8, 9, 10]),
               'instances': [{'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}]}
            ]
        """

        collection = []
        for routine in self.__routines:
            collection.append({
                'centroid': routine.centroid,
                'instances': routine.get_sequences().to_collection()
            })
        return collection

    def is_empty(self) -> bool:
        """
        Returns `True` if the collection is empty, `False` otherwise

        Returns:
            `bool`. `True` if the collection is empty, `False` otherwise

        Examples:
            >>> routines = Routines()
            >>> routines.is_empty()
            True

            >>> cluster = Cluster(np.array([3, 4, 5, 6]), Sequence(Subsequence(np.array([1, 2, 3, 4]), datetime.date(2021, 1, 1), 0))
            >>> routines.add_routine(cluster)
            >>> routines.is_empty()
            False
        """

        return len(self.__routines) == 0
