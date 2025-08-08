from abc import ABC
from abc import abstractmethod


class CacheBase(ABC):
    @abstractmethod
    def get_record(self, *args, **kwargs):
        """Retrieve a single record from the cache."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def get_records(self, *args, **kwargs):
        """Retrieve multiple records from the cache."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def add_records(self, *args, **kwargs):
        """Add records to the cache."""
        raise NotImplementedError("This method should be implemented by subclasses.")
