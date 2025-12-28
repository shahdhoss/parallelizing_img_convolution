from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, input_file_name
from pyspark.sql.types import *
import grpc
import base64
import json

# Import gRPC stubs
import imageConvolution_pb2
import imageConvolution_pb2_grpc


class GrpcImageClient:    
    def __init__(self, servers=['localhost:50051', 'localhost:50052']):
        self.servers = servers
        self.current_idx = 0
        self.max_retries = 3
    
    def process_image(self, image_bytes, filename):
        
        for attempt in range(self.max_retries):
            server_idx = (self.current_idx + attempt) % len(self.servers)
            server = self.servers[server_idx]
            
            try:
                # Create gRPC channel
                channel = grpc.insecure_channel(server)
                stub = imageConvolution_pb2_grpc.ImageConvolutionStub(channel)
                
                kernel = [0, 1, 0, -1, 5, -1, 0, -1, 0]
                chunk = imageConvolution_pb2.ImageChunk(
                    chunk_data=image_bytes,
                    chunk_index=0,
                    total_chunks=1,
                    kernel=kernel
                )
                
                def request_gen():
                    yield chunk
                
                responses = stub.StreamConvolution(request_gen(), timeout=5.0)
                
                for response in responses:
                    if response.success:
                        channel.close()
                        return {
                            'success': True,
                            'server': server,
                            'filename': filename
                        }
                
            except Exception as e:
                print(f"⚠ Attempt {attempt + 1} failed: {e}")
        
        return {'success': False, 'error': 'All retries failed'}


_grpc_client = None

def get_grpc_client():
    global _grpc_client
    if _grpc_client is None:
        _grpc_client = GrpcImageClient()
    return _grpc_client


def process_image(image_bytes, filename):
    """ process image via gRPC"""
    try:
        client = get_grpc_client()
        result = client.process_image(image_bytes, filename)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)})


def main():
    
    print("SPARK FILE STREAMING")
    spark = SparkSession.builder \
        .appName("ImageStreaminga") \
        .master("local[4]") \
        .config("spark.sql.streaming.checkpointLocation", "/tmp/checkpoint") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    print("✓ Spark session created\n")
    
    print("📂 Monitoring directory: ../../data/img/")
    print("   Spark will process any new images placed here\n")
    
    file_stream = spark \
        .readStream \
        .format("binaryFile") \
        .option("pathGlobFilter", "*.jpg") \
        .load("../../data/img/")
    
    process_udf = udf(process_image, StringType())
    

    processed_stream = file_stream \
        .withColumn("filename", input_file_name()) \
        .withColumn("result", process_udf(col("content"), col("filename")))
    
    # Parse result JSON
    result_schema = StructType([
        StructField("success", BooleanType()),
        StructField("server", StringType()),
        StructField("error", StringType())
    ])
    
    output_stream = processed_stream \
        .withColumn("parsed_result", from_json(col("result"), result_schema)) \
        .select(
            col("filename"),
            col("parsed_result.success").alias("success"),
            col("parsed_result.server").alias("server"),
            col("parsed_result.error").alias("error")
        )
    
    

    print("  Micro-batch trigger: 5 seconds")
    
    query = output_stream \
        .writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", False) \
        .trigger(processingTime="5 seconds") \
        .start()
    
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n\n⏹ Stopping stream...")
        query.stop()
        spark.stop()


if __name__ == '__main__':
    main()
