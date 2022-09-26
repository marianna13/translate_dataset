from pyspark.sql import SparkSession
import pyspark.sql.functions as F
def main():
  spark = SparkSession.builder.config("spark.driver.memory", "500G").config("spark.local.dir", "/scratch/marianna").master("local[96]").appName('spark-stats').getOrCreate() 

  root_path = "/scratch/marianna/FULL"

  datasets = [(False, True)]

  for (multilingual, nolang) in datasets:
    path = root_path
    df = spark.read.parquet(path)
    df = df.withColumnRenamed("WIDTH", "h")
    df = df.withColumnRenamed("HEIGHT", "w")
    df = df.withColumnRenamed("h", "HEIGHT")
    df = df.withColumnRenamed("w", "WIDTH")
    df = df.drop('h', 'w')

    def nm(n):  
      m = int(n / (10**6))
      return str(m) + "M ("+str(n)+")"

    print("Number of uniques", nm(df.count()))

    size_range = [[0,128], [128,256],[256,512], [512, 1024], [1024, None]]
    text_range = [[0,25], [25,50],[50,100], [100,150], [150, None]]

    def compute_range_stats(field_name, ranges):
      selected = df.select(field_name).persist()
      for r in ranges:
        if r[1] is None:
          print("Number with "+field_name+" >=", r[0], nm(selected[(selected[field_name] >= r[0])].count()))
        else:
          print("Number with "+field_name+" >=", r[0], "and "+field_name+" <=", r[1], \
            nm(selected[(selected[field_name] >= r[0]) & (selected[field_name] <= r[1])].count()))

    df = df.withColumn("lenengtext", F.length("ENG TEXT"))
    compute_range_stats("WIDTH", size_range)
    compute_range_stats("HEIGHT", size_range)
    compute_range_stats("lenengtext", text_range)

  
main()
