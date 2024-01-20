"""This module defines project-level constants."""

aws_region: str = "us-east-1"
aws_service_quota_for_general_type_instances: dict = {"ServiceCode": "ec2", "QuotaCode": "L-1216C47A"} # AWS Service Quota code for Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances
asd_headscale_container_image: str = 'johncarvalho/asd-dev:headscale1.3'
ec2_ebs_size: int = 150 # Default value (user will still be able to adjust)
ec2_asg_ami_id: str = 'ami-09504161962d97cef'