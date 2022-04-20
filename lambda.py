"""
1. serializeImageData
"""
import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    key = event["s3_key"]
    bucket = event["s3_bucket"]
    file_name = '/tmp/image.png'
    s3.download_file(bucket, key, file_name)
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

"""
2. invokingEndpoint
"""

import json
import base64
import boto3
from sagemaker.serializers import IdentitySerializer

# Name of the deployed model
ENDPOINT = "image-classification-2022-04-20-09-06-00-803"
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    """A function to invoke endpoint"""
    image = base64.b64decode(event['image_data'])
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType='application/x-image', Body=image)
    inferences = response['Body'].read().decode('utf-8')
    event["inferences"] = [float(x) for x in inferences[1:-1].split(',')] 
    return {
        'statusCode': 200,
        'body': {
            "image_data": event['image_data'],
            "s3_bucket": event['s3_bucket'],
            "s3_key": event['s3_key'],
            "inferences": event['inferences'],
        }
    }


"""
3. filterResults
"""
import json
THRESHOLD = .93

def lambda_handler(event, context):
    """A function to filter low quality predictions"""
    event_in = json.loads(event['body'])
    inferences = event_in['inferences']
    meets_threshold = any (x >= THRESHOLD for x in inferences)
    if meets_threshold:
        return {
        'statusCode': 200,
        'body': {
            "image_data": event_in['body']['image_data'],
            "s3_bucket": event_in['body']['s3_bucket'],
            "s3_key": event_in['body']['s3_key'],
            "inferences": inferences,
            }
        }
    else:
        raise ValueError(f"low threshold: {inferences}")

