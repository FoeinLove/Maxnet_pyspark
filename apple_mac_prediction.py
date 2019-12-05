from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.ml.classification import RandomForestClassificationModel
import time 

sc = SparkContext()
spark = SparkSession(sc)

i = time.strftime('%Y%m%d',time.localtime(time.time()))

try:
    model = RandomForestClassificationModel.load('/data/user/hive/warehouse/ian/model/mac_id_12_rf_cv_tuning_model')

    d1 = spark.read.csv('/user/maxnet/database/sig.db/data_visual_macinfo_res',sep='\x01').select('_c0','_c4','_c5').distinct().withColumnRenamed('_c0','mac').withColumnRenamed('_c4','manu').withColumnRenamed('_c5','prior')
    d2 = d1.filter(d1.prior < 4)
    d3 = d2.filter(d2.manu == '苹果').select('mac')

    df = d3.withColumn('m1',substring(d3.mac,1,1))           .withColumn('m2',substring(d3.mac,2,1))           .withColumn('m3',substring(d3.mac,3,1))           .withColumn('m4',substring(d3.mac,4,1))           .withColumn('m5',substring(d3.mac,5,1))           .withColumn('m6',substring(d3.mac,6,1))           .withColumn('m7',substring(d3.mac,7,1))           .withColumn('m8',substring(d3.mac,8,1))           .withColumn('m9',substring(d3.mac,9,1))           .withColumn('m10',substring(d3.mac,10,1))           .withColumn('m11',substring(d3.mac,11,1))           .withColumn('m12',substring(d3.mac,12,1))  

    df = df.withColumn('f1',conv(df.m1, 16, 10)).withColumn('f2',conv(df.m2, 16, 10))                         .withColumn('f3',conv(df.m3, 16, 10)).withColumn('f4',conv(df.m4, 16, 10))                         .withColumn('f5',conv(df.m5, 16, 10)).withColumn('f6',conv(df.m6, 16, 10))                         .withColumn('f7',conv(df.m7, 16, 10)).withColumn('f8',conv(df.m8, 16, 10))                         .withColumn('f9',conv(df.m9, 16, 10)).withColumn('f10',conv(df.m10, 16, 10))                         .withColumn('f11',conv(df.m11, 16, 10)).withColumn('f12',conv(df.m12, 16, 10))

    df = df.select('mac',                col('f1').cast('float'),col('f2').cast('float'),                col('f3').cast('float'),col('f4').cast('float'),                col('f5').cast('float'),col('f6').cast('float'),                col('f7').cast('float'),col('f8').cast('float'),                col('f9').cast('float'),col('f10').cast('float'),                col('f11').cast('float'),col('f12').cast('float'))

    from pyspark.ml.feature import VectorAssembler
    vec = VectorAssembler(inputCols=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12'],outputCol='features')
    df = vec.transform(df)

    result = model.transform(df)
    mapping_table = spark.read.parquet('/data/user/hive/warehouse/ian/feature/mapping_table').withColumnRenamed('id','predict_id')
    result = result.select('mac','prediction')
    tmp = result.join(mapping_table,result.prediction == mapping_table.label)
    prediction = tmp.select('mac','predict_id').distinct()

    prediction.write.saveAsTable('sig.apple_mac_prediction_%s'%(i), None, "overwrite", None)
    
except:
    pass

