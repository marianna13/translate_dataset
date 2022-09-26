from pyspark.sql import SparkSession
import os
from pyspark.sql.functions import rand

spark = SparkSession.builder.config("spark.driver.memory", "600G").config("spark.sql.files.ignoreCorruptFiles", "true").config("spark.local.dir", "/scratch/marianna").master("local[96]").appName('spark-stats').getOrCreate()


file_path_list = os.listdir('/scratch/main')
additional = os.listdir('/scratch/additiona')
additional = [f'/scratch/additiona/{f}' for f in additional]
file_path_list = [f'/scratch/main/{f}' for f in file_path_list]

df1 = spark.read.parquet(*file_path_list)
df2 = spark.read.parquet(*additional)
df = df1.union(df2) 
df = df.drop_duplicates(['URL', 'TEXT'])
df = df.orderBy(rand()) 

df = df.repartition(128)
df.write.parquet('/scratch/marianna/FULL')
