"""
Data Structures.

This script defines the data structures needed for the algorithm of routine detection.

The module contains the following public classes

Subsequence: Basic data structure.
    Parameters:
        * instance: np.ndarray, sequence of numbers from the time series
        * date: datetime.date, the timestamp where the instance belongs
        * starting_point: int, the temporary starting point of the subsequence

    Public methods:
        * get_instance
        * get_date
        * get_starting_point
        * to_collection
        * Magnitude
        * Distance


Sequence: Structure based on a list of Subsequences. It doesn't take parameters.
    Public Methods:
        * add_sequence
        * get_by_starting_point
        * set_by_starting_point
        * get_starting_points
        * get_dates
        * get_instances
        * to_collection

Cluster: Represents the structure of a cluster from a centroid and a Sequence associated to an instance from the centroid
    Parameters:
        * centroid: np.ndarray, the centroid of the cluster
        * instances: Sequence, instances which belongs to the cluster

    Public Methods
"""
import numpy as np
import datetime
from typing import Union


class Subsequence:
    def __init__(self, instance: np.ndarray, date: datetime.date, starting_point: int) -> None:
        self.__checkType(instance, date, starting_point)
        self.__instance = instance
        self.__date = date
        self.__starting_point = starting_point

    @staticmethod
    def __checkType(instance: np.ndarray, date: datetime.date, starting_point: int) -> None:
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
        return self.__instance

    def get_date(self) -> datetime.date:
        return self.__date

    def get_starting_point(self) -> int:
        return self.__starting_point

    def to_collection(self) -> dict:
        return {"instance": self.__instance, "date": self.__date, "starting_point": self.__starting_point}

    def Magnitude(self) -> float:
        return np.max(self.__instance)

    def Distance(self, other: Union['Subsequence', np.ndarray]) -> float:
        if isinstance(other, np.ndarray):
            return np.max(np.abs(self.__instance - other))

        if isinstance(other, Subsequence):
            return np.max(np.abs(self.__instance - other.get_instance()))

        raise TypeError("other must be an instance of ndarray or a Subsequence")


class Sequence:
    def __init__(self, subsequence: Union[Subsequence, None] = None):
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
        if not isinstance(new, Subsequence):
            raise TypeError("new has to be an instance of Subsequence")

        if self._alreadyExists(new):
            raise RuntimeError("new sequence already exists ")

        self.__list_sequences.append(new)

    def get_by_starting_point(self, starting_point: int) -> Union['Subsequence', None]:
        for subseq in self.__list_sequences:
            if subseq.get_starting_point() == starting_point:
                return subseq
        return None

    def set_by_starting_point(self, starting_point: int, new_sequence: 'Subsequence') -> None:
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

    def get_starting_points(self) -> list:
        return [subseq.get_starting_point() for subseq in self.__list_sequences]

    def get_dates(self) -> list:
        return [subseq.get_date() for subseq in self.__list_sequences]

    def get_subsequences(self) -> list[np.ndarray]:
        return [subseq.get_instance() for subseq in self.__list_sequences]

    def to_collection(self) -> list[dict]:
        collection = []
        for subseq in self.__list_sequences:
            collection.append({
                'instance': subseq.get_instance(),
                'date': subseq.get_date(),
                'starting_point': subseq.get_starting_point()
            })
        return collection


class Cluster:
    def __init__(self, centroid: np.ndarray, instances: 'Sequence'):
        self.__centroid = centroid
        self.__instances = instances

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
        if not isinstance(new_instance, Subsequence):
            raise TypeError("new sequence must be an instance of Subsequence")

        if self.__instances._alreadyExists(new_instance):
            raise ValueError("new sequence is already an instance of the cluster")

        self.__instances.add_sequence(new_instance)

    def get_sequences(self) -> 'Sequence':
        return self.__instances

    def update_centroid(self) -> None:
        self.__centroid = np.mean(self.__instances.get_subsequences(), axis=0)

    @property
    def centroid(self) -> np.ndarray:
        return self.__centroid

    @centroid.setter
    def centroid(self, subsequence: 'Subsequence') -> None:
        if not isinstance(subsequence, Subsequence):
            raise TypeError("Must be passed an instance of Subsequence to set the value of the centroid")

        self.__centroid = subsequence.get_instance()

    def get_starting_points(self) -> list:
        return self.__instances.get_starting_points()

    def get_dates(self) -> list:
        return self.__instances.get_dates()


class Routines:
    def __init__(self, cluster: Union[Cluster, None] = None):
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
        if not isinstance(new_routine, Cluster):
            raise TypeError("new_routine has to be an instance of Cluster")

        self.__routines.append(new_routine)

    def drop_indexes(self, to_drop: list[int]) -> 'Routines':
        new_routines = Routines()
        for idx, cluster in enumerate(self.__routines):
            if idx not in to_drop:
                new_routines.add_routine(cluster)
        return new_routines

    def get_routines(self) -> list:
        return self.__routines

    def to_collection(self) -> list:
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
