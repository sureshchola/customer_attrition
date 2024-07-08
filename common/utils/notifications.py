import boto3

def job_status(pipeline,state,account,region,env,body=''):

    """
    args:
    pipeline - pipeline stage either training or inferencing
    state - state of pipeline either succeeded or failed
    account - account number
    region - region of aws
    env - environment
    body - email body
    
    returns:
    None
    """

    client = boto3.client('sns',region_name=region)
    x=client.publish(
              TargetArn =f"arn:aws:sns:{region}:{account}:job_notifications",
              Message = f"{pipeline} stage for environment {env} has {state}.\n{body}",
              Subject = f"{pipeline} stage for environment {env} - {state}")
    return x