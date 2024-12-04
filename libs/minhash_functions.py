from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType
import random


class MinHashSpark:
    """
    A class to compute MinHash signatures using Spark DataFrames.
    """
    def __init__(self, num_hashes=100, prime=100003):
        """
        Initialize the MinHashSpark object.

        :param num_hashes: Number of hash functions to use.
        :param prime: A prime number larger than the largest possible input for hashing.
        """
        self.num_hashes = num_hashes
        self.prime = prime
        self.hash_functions = self._generate_hash_functions()

    def _generate_hash_functions(self):
        """
        Generate random hash functions of the form h(x) = (a * x + b) % p.
        """
        hash_functions = []
        for _ in range(self.num_hashes):
            a = random.randint(1, self.prime - 1)
            b = random.randint(0, self.prime - 1)
            hash_functions.append((a, b))
        return hash_functions

    def compute_signature_udf(self):
        """
        Create a PySpark UDF to compute MinHash signatures for a list of movie IDs.
        """
        def compute_signature(movies):
            signature = []
            for a, b in self.hash_functions:
                min_hash = min(((a * movie_id + b) % self.prime) for movie_id in movies)
                signature.append(min_hash)
            return signature

        return F.udf(compute_signature, ArrayType(IntegerType()))

    def jaccard_similarity(self, sig1, sig2):
        """
        Compute Jaccard similarity between two MinHash signatures.
        """
        return sum(1 for x, y in zip(sig1, sig2) if x == y) / len(sig1)
