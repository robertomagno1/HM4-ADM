from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, to_date, year, explode, split
import pandas as pd

def initialize_spark(app_name="MovieLens Analysis"):
    """
    Initialize the Spark session.
    """
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_data(spark, path):
    """
    Load all datasets into Spark DataFrames.
    """
    tag_df = spark.read.csv(f"{path}/tag.csv", header=True, inferSchema=True)
    rating_df = spark.read.csv(f"{path}/rating.csv", header=True, inferSchema=True)
    movie_df = spark.read.csv(f"{path}/movie.csv", header=True, inferSchema=True)
    link_df = spark.read.csv(f"{path}/link.csv", header=True, inferSchema=True)
    genome_scores_df = spark.read.csv(f"{path}/genome_scores.csv", header=True, inferSchema=True)
    genome_tags_df = spark.read.csv(f"{path}/genome_tags.csv", header=True, inferSchema=True)
    
    return tag_df, rating_df, movie_df, link_df, genome_scores_df, genome_tags_df

def inspect_data(df, num_rows=5, convert_to_pandas=False):
    """
    Inspect the content of a Spark DataFrame.
    - df: The Spark DataFrame to inspect.
    - num_rows: Number of rows to display or retrieve.
    - convert_to_pandas: If True, converts the DataFrame to Pandas.
    """
    if convert_to_pandas:
        return df.limit(num_rows).toPandas()
    else:
        df.show(num_rows)

def preprocess_movies_ratings(movie_df, rating_df):
    """
    Preprocess the movies and ratings data by joining and extracting genres and year.
    """
    # Join movies and ratings
    movies_ratings_df = movie_df.join(rating_df, "movieId")
    
    # Extract year from timestamp
    movies_ratings_df = movies_ratings_df.withColumn("year", year(to_date(col("timestamp"))))
    
    # Explode genres into multiple rows
    movies_ratings_df = movies_ratings_df.withColumn("genre", explode(split(col("genres"), "\\|")))
    
    return movies_ratings_df

def calculate_avg_ratings_by_genre(movies_ratings_df):
    """
    Calculate average ratings and count of ratings by genre.
    """
    return movies_ratings_df.groupBy("genre").agg(
        avg("rating").alias("average_rating"),
        count("rating").alias("rating_count")
    ).orderBy(col("average_rating").desc())

def calculate_ratings_by_year(movies_ratings_df):
    """
    Calculate average ratings and count of ratings by year.
    """
    return movies_ratings_df.groupBy("year").agg(
        avg("rating").alias("average_rating"),
        count("rating").alias("rating_count")
    ).orderBy("year")

def save_to_csv(dataframe, path):
    """
    Save the Spark DataFrame to a CSV file.
    """
    dataframe.coalesce(1).write.csv(path, header=True)

def merge_datasets(tag_df, rating_df, movie_df, link_df, genome_scores_df, genome_tags_df):
    # Rename ambiguous columns
    tag_df = tag_df.withColumnRenamed("tag", "user_tag").withColumnRenamed("timestamp", "tag_timestamp")
    rating_df = rating_df.withColumnRenamed("timestamp", "rating_timestamp")
    genome_tags_df = genome_tags_df.withColumnRenamed("tag", "genome_tag")

    # Merge ratings with movies
    user_movie_df = movie_df.join(rating_df, "movieId", "inner")
    
    # Merge with tags
    user_movie_df = user_movie_df.join(tag_df, ["userId", "movieId"], "left")
    
    # Merge genome scores with genome tags
    genome_df = genome_scores_df.join(genome_tags_df, "tagId", "inner")
    
    # Merge genome data with main dataset
    final_df = user_movie_df.join(genome_df, "movieId", "left")
    
    # Handle missing values
    final_df = final_df.fillna({
        "user_tag": "No Tag",
        "genres": "Unknown",
        "relevance": 0.0
    })

    # Add a date column for easier handling
    final_df = final_df.withColumn("rating_date", to_date(col("rating_timestamp")))

    return final_df