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
from typing import Union


class Subsequence:
    """
    Basic data structure.

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
        :param instance: np.ndarray, the instance of the subsequence
        :param date: datetime.date, the date of the subsequence
        :param starting_point: int, the starting point of the subsequence
        """
        self.__checkType(instance, date, starting_point)
        self.__instance = instance
        self.__date = date
        self.__starting_point = starting_point

    @staticmethod
    def __checkType(instance: np.ndarray, date: datetime.date, starting_point: int) -> None:
        """
        Check the type of the parameters

        :param instance: np.ndarray, the instance of the subsequence
        :param date: datetime.date, the date of the subsequence
        :param starting_point: int, the starting point of the subsequence

        :raises TypeError: if the parameters are not of the correct type
        """

        err_inst = "Instances must be an arrays"
        err_date = "Date must be a timestamps"
        err_stpoint = "starting_point must be a integer"

        if not isinstance(instance, np.ndarray):
            raise TypeError(err_inst)

        if not isinstance(date, datetime.date):
            raise TypeError(err_date)

        if not isinstance(starting_point, int):
            raise TypeError(err_stpoint)

    def __repr__(self):
        return f"Subsequence(\n\t instance={self.__instance} \n\t date={self.__date} \n\t starting point = {self.__starting_point}\n)"

    def __str__(self):
        return f"Subsequence(\n\t instances={self.__instance} \n\t date={self.__date} \n\t starting point = {self.__starting_point}\n)"

    def __len__(self):
        return len(self.__instance)

    def __getitem__(self, index: int):
        return self.__instance[index]

    def __eq__(self, other: 'Subsequence') -> bool:
        if not np.array_equal(self.__instance, other.get_instance()):
            return False
        if self.__date != other.get_date() or self.__starting_point != other.get_starting_point():
            return False

        return True

    def get_instance(self) -> np.ndarray:
        """
        Returns the instance of the subsequence
        :return: np.ndarray. The instance of the subsequence
        """
        return self.__instance

    def get_date(self) -> datetime.date:
        """
        Returns the date of the subsequence
        :return: datetime.date. The date of the subsequence
        """
        return self.__date

    def get_starting_point(self) -> int:
        """
        Returns the starting point of the subsequence
        :return: int. The starting point of the subsequence
        """

        return self.__starting_point

    def to_collection(self) -> dict:
        """
        Returns the subsequence as a dictionary
        :return: dict. The subsequence as a dictionary
        """

        return {"instance": self.__instance, "date": self.__date, "starting_point": self.__starting_point}

    def Magnitude(self) -> float:
        """
        Returns the magnitude of the subsequence
        :return: np.max. The magnitude of the subsequence
        """
        return np.max(self.__instance)

    def Distance(self, other: Union['Subsequence', np.ndarray]) -> float:
        """
        Returns the distance between the subsequence and another subsequence or array
        :param other: Union['Subsequence', np.ndarray]. The subsequence or array to compare

        :return: np.max. The distance between the subsequence and another subsequence or array
        :raises TypeError: if the parameter is not of the correct type
        """
        if isinstance(other, np.ndarray):
            return np.max(np.abs(self.__instance - other))

        if isinstance(other, Subsequence):
            return np.max(np.abs(self.__instance - other.get_instance()))

        raise TypeError("other must be an instance of ndarray or a Subsequence")


class Sequence:
    """
    Represents a sequence of subsequences

    Parameters:
        * subsequence: Union[Subsequence, None], the subsequence to add to the sequence

    Public Methods:
        * add_sequence: adds a subsequence to the sequence
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

    def __init__(self, subsequence: Union[Subsequence, None] = None):
        """
        :param subsequence: Union[Subsequence, None], the subsequence to add to the sequence
        :raises TypeError: if the parameter is not of the correct type
        """
        if subsequence is not None:
            if not isinstance(subsequence, Subsequence):
                raise TypeError("subsequence has to be an instance of Subsequence")

            self.__list_sequences = [subsequence]
        else:
            self.__list_sequences = []

    def __repr__(self):
        out_string = "Sequence(\n\tlist_sequences=[\n"
        for seq in self.__list_sequences:
            out_string += f" {seq},\n"

        out_string = out_string[:-2] + out_string[-1] + "]"
        return out_string

    def __str__(self):
        out_string = "Sequence(\n\tlist_sequences=[\n"
        for seq in self.__list_sequences:
            out_string += f" {seq},\n"

        out_string = out_string[:-2] + out_string[-1] + "]"
        return out_string

    def __len__(self) -> int:
        return len(self.__list_sequences)

    def __getitem__(self, index: int) -> 'Subsequence':
        return self.__list_sequences[index]

    def __setitem__(self, index: int, value: 'Subsequence') -> None:
        self.__list_sequences[index] = value

    def __iter__(self):
        return iter(self.__list_sequences)

    def __contains__(self, item: 'Subsequence') -> bool:
        return item in self.__list_sequences

    def __delitem__(self, index: int) -> None:
        del self.__list_sequences[index]

    def __add__(self, other: 'Sequence') -> 'Sequence':
        new_sequence = Sequence()
        new_sequence.__list_sequences = self.__list_sequences + other.__list_sequences
        return new_sequence

    def _alreadyExists(self, subsequence: 'Subsequence') -> bool:
        """
        Check if the subsequence already exists in the sequence
        :param subsequence: Subsequence. The subsequence to check
        :return: bool. True if the subsequence already exists, False otherwise
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
        :param new: Subsequence. The subsequence to add
        :raises TypeError: if the parameter is not of the correct type
        :raises RuntimeError: if the subsequence already exists
        """
        if not isinstance(new, Subsequence):
            raise TypeError("new has to be an instance of Subsequence")

        if self._alreadyExists(new):
            raise RuntimeError("new sequence already exists ")

        self.__list_sequences.append(new)

    def get_by_starting_point(self, starting_point: int) -> Union['Subsequence', None]:
        """
        Returns the subsequence with the specified starting point
        :param starting_point: int. The starting point of the subsequence
        :return: Union[Subsequence, None]. The subsequence with the specified starting point if it exists. Otherwise, None
        """
        for subseq in self.__list_sequences:
            if subseq.get_starting_point() == starting_point:
                return subseq
        return None

    def set_by_starting_point(self, starting_point: int, new_sequence: 'Subsequence') -> None:
        """
        Sets the subsequence with the specified starting point
        :param starting_point: int. The starting point of the subsequence
        :param new_sequence: Subsequence. The new subsequence
        :raises TypeError: if the parameter is not of the correct type
        :raises ValueError: if the starting point does not exist
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
        :return: list[int]. The starting points of the subsequences
        """
        return [subseq.get_starting_point() for subseq in self.__list_sequences]

    def get_dates(self) -> list[datetime.date]:
        """
        Returns the dates of the subsequences
        :return: list[datetime.date]. The dates of the subsequences
        """
        return [subseq.get_date() for subseq in self.__list_sequences]

    def get_subsequences(self) -> list[np.ndarray]:
        """
        Returns the instances of the subsequences
        :return: list[np.ndarray]. The instances of the subsequences
        """
        return [subseq.get_instance() for subseq in self.__list_sequences]

    def to_collection(self) -> list[dict]:
        """
        Returns the sequence as a list of dictionaries
        :return: list[dict]. The sequence as a list of dictionaries
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
        * centroid: np.ndarray, the centroid of the cluster
        * instances: Sequence, the sequence of subsequences

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
        >>> cluster = Cluster(np.array([3, 4, 5, 6]), sequence)
        >>> cluster.get_sequences().to_collection()
        [{'instance': np.array([1, 2, 3, 4]), 'date': datetime.date(2021, 1, 1), 'starting_point': 0},
         {'instance': np.array([5, 6, 7, 8]), 'date': datetime.date(2021, 1, 2), 'starting_point': 4}]
        >>> cluster.get_starting_points()
        [0, 4]
        >>> cluster.get_dates()
        [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)]
        >>> cluster.centroid
        np.array([3, 4, 5, 6])
        >>> cluster.centroid = np.array([1, 2, 3, 4])
        >>> cluster.centroid
        np.array([1, 2, 3, 4])
        >>> cluster.update_centroid()
        >>> cluster.centroid
        np.array([3, 4, 5, 6])
    """

    def __init__(self, centroid: np.ndarray, instances: 'Sequence'):
        self.__centroid: np.ndarray = centroid
        self.__instances: Sequence = instances

    def __str__(self):
        out_string = f"Cluster(\n\t -Centroid: {self.__centroid} \n"
        out_string += f"\t -Instances: {[instance for instance in self.__instances.get_subsequences()]}\n"
        out_string += f"\t -Dates: {[date for date in self.__instances.get_dates()]}\n"
        out_string += f"\t -Starting Points: {[sp for sp in self.__instances.get_starting_points()]}\n)"
        return out_string

    def __repr__(self):
        out_string = f"Cluster(\n\t -Centroid: {self.__centroid} \n"
        out_string += f"\t -Instances: {[instance for instance in self.__instances.get_subsequences()]}\n"
        out_string += f"\t -Dates: {[date for date in self.__instances.get_dates()]}\n"
        out_string += f"\t -Starting Points: {[sp for sp in self.__instances.get_starting_points()]}\n)"
        return out_string

    def __del__(self):
        del self.__centroid
        del self.__instances

    def __len__(self) -> int:
        return len(self.__instances)

    def __getitem__(self, index: int) -> 'Subsequence':
        return self.__instances[index]

    def __setitem__(self, index: int, value: 'Subsequence') -> None:
        self.__instances[index] = value

    def __iter__(self):
        return iter(self.__instances)

    def __contains__(self, item: 'Subsequence') -> bool:
        return item in self.__instances

    def __delitem__(self, index: int) -> None:
        del self.__instances[index]

    def __add__(self, other: 'Cluster') -> 'Cluster':
        new_instances = self.__instances + other.get_sequences()
        new_centroid = np.mean(new_instances.get_subsequences(), axis=0)
        return Cluster(centroid=new_centroid, instances=new_instances)

    def add_instance(self, new_instance: 'Subsequence') -> None:
        """
        Adds a subsequence to the instances of the cluster
        :param new_instance: Subsequence. The subsequence to add
        :raises TypeError: if the parameter is not of the correct type
        :raises ValueError: if the subsequence is already an instance of the cluster
        """
        if not isinstance(new_instance, Subsequence):
            raise TypeError("new sequence must be an instance of Subsequence")

        if self.__instances._alreadyExists(new_instance):
            raise ValueError("new sequence is already an instance of the cluster")

        self.__instances.add_sequence(new_instance)

    def get_sequences(self) -> 'Sequence':
        """
        Returns the sequence of the cluster
        :return: Sequence. The sequence of the cluster
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
        :param subsequence: Union[Subsequence|np.ndarray]. The subsequence to set as the centroid
        :raises TypeError: if the parameter is not a Subsequence or a numpy array
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
        :return: list[int]. The starting points of the subsequences
        """
        return self.__instances.get_starting_points()

    def get_dates(self) -> list[datetime.date]:
        """
        Returns the dates of the subsequences
        :return: list[datetime.date]. The dates of the subsequences
        """
        return self.__instances.get_dates()

    def cumulative_magnitude(self) -> float | int:
        """
        Returns the magnitude's sum of the subsequences that belongs to the instances within the cluster

        :return: `float`. The magnitude's sum of the subsequences
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

    def __init__(self, cluster: Union[Cluster, None] = None):
        """
        :param cluster: Union[Cluster, None], the cluster to add to the collection
        :raises TypeError: if the parameter is not of the correct type
        """
        if cluster is not None:
            if not isinstance(cluster, Cluster):
                raise TypeError("cluster has to be an instance of Cluster")

            self.__routines = [cluster]
        else:
            self.__routines = []

    def __repr__(self):
        out_string = "Routines(\n\tlist_routines=[[\n"
        for routine in self.__routines:
            out_string += f" {routine},\n"

        out_string = out_string[:-2] + out_string[-1] + "])"
        return out_string

    def __str__(self):
        out_string = "Routines(\n\tlist_routines=[[\n"
        for routine in self.__routines:
            out_string += f" {routine},\n"

        out_string = out_string[:-2] + out_string[-1] + "])"
        return out_string

    def add_routine(self, new_routine: 'Cluster') -> None:
        """
        Adds a cluster to the collection

        :param new_routine: Cluster. The cluster to add
        :raises TypeError: if the parameter is not of the correct type

        """
        if not isinstance(new_routine, Cluster):
            raise TypeError("new_routine has to be an instance of Cluster")

        self.__routines.append(new_routine)

    def drop_indexes(self, to_drop: list[int]) -> 'Routines':
        """
        Drops the clusters with the specified indexes

        :param to_drop: list[int]. The indexes of the clusters to drop
        :return: Routines. The collection without the dropped clusters
        """

        new_routines = Routines()
        for idx, cluster in enumerate(self.__routines):
            if idx not in to_drop:
                new_routines.add_routine(cluster)
        return new_routines

    def get_routines(self) -> list[Cluster]:
        """
        Returns the clusters of the collection
        :return: list. Returns all the clusters of the routines
        """

        return self.__routines

    def to_collection(self) -> list[dict]:
        """
        Returns the collection as a list of dictionaries
        :return: list[dict]. The collection as a list of dictionaries
        """
        collection = []
        for routine in self.__routines:
            collection.append({
                'centroid': routine.centroid,
                'instances': routine.get_sequences().to_collection()
            })
        return collection

    def __len__(self) -> int:
        return len(self.__routines)

    def __getitem__(self, index: int) -> 'Cluster':
        return self.__routines[index]

    def __setitem__(self, index: int, value: 'Cluster') -> None:
        self.__routines[index] = value

    def __iter__(self):
        return iter(self.__routines)

    def __contains__(self, item: 'Cluster') -> bool:
        return item in self.__routines

    def __delitem__(self, index: int) -> None:
        del self.__routines[index]

    def __add__(self, other: 'Routines') -> 'Routines':
        new_routines = Routines()
        new_routines.__routines = self.__routines + other.__routines
        return new_routines
