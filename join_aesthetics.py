from pyspark.sql import SparkSession
import os


spark = SparkSession.builder.config("spark.driver.memory", "500G").config("spark.sql.files.ignoreCorruptFiles", "true").config("spark.local.dir", "/scratch/marianna").master("local[96]").appName('spark-stats').getOrCreate()


file_path_list = os.listdir('/scratch/marianna/FULL')
file_path_list = [f'/scratch/marianna/FULL/{f}' for f in file_path_list]

ast_path_list = os.listdir('/fsx/marianna/translate_dataset/aesthetics')
ast_path_list = [f'/fsx/marianna/translate_dataset/aesthetics/{f}' for f in ast_path_list]

df = spark.read.parquet(*file_path_list)
ast_df = spark.read.parquet(*ast_path_list)
ast_df = ast_df.select("prediction", "hash")

joined_df = df.join(ast_df, ["hash"], 'left') 
joined_df = joined_df.repartition(128)

joined_df.write.parquet("/scratch/marianna/JOINED")
