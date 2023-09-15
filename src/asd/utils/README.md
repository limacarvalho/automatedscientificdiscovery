# ASD - Utils
Resource management and interaction with AWS-related operations, local system operations, and Ray cluster operations, both locally and on AWS.

## Modules:

1. `aws_utils.py`:
    - Manages AWS-related operations, primarily focusing on Step Functions.
    - Manages AWS infrastructure components.
    - Interacts with AWS Boto3 library for various AWS services.

2. `os_utils.py`:
    - Provides utility functions for system-level operations.
    - Supports directory deletion, AWS credentials management, and more.

3. `aws_infrastructure.py`:
    - Sets up and manages AWS infrastructure.
    - Interacts with AWS services, such as StepFunctions and CloudFormation.

4. `ray_cluster.py`:
    - Manages Ray Cluster for local and remote operations.
    - Provides functionality to check the status, start, stop, and purge Ray clusters.

## Features:

- Comprehensive logging for easy debugging and traceability.
- Robust error handling to ensure reliable execution.
- Flexibility to work with local or AWS remote operations.
- Clear modular structure for easy maintainability.

## How to Use:

1. Ensure you have the required Python libraries installed (`boto3`, `psutil`, `pytz`).
2. Ensure AWS credentials are set up if working with AWS components.
3. Use the respective classes and methods for the desired operations, as documented in each module.