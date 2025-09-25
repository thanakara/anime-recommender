## AWS SageMaker setup

Use an IAM user account and *not* the root user.\
After creating an IAM Execution Role, visit AWS SageMaker AI.

The new Dashboard feature let us know which instances are currenly InService, for monitoring and also avoid unnecessary pricing. After creating a Domain, we visit Notebook instances to create one.

Using our Execution Role, select an instance type of `ml.m5.xlarge` or for better balance the `ml.m5.2xlarge`. The latter consists of $4$ vCPU and $32$ GiB of memory which suffices for this project. For more details click [here](https://aws.amazon.com/ec2/instance-types/m5/).

Attaching an EBS volume to the Notebook instance is needed. Since we are dealing with a pretty big dataset, $30$ GiB is more than enough.

Additionally, under Git Repocity, attach this repository by cloning.

After the Notebook is ready,make sure you are in the root directory.
Start the Notebook instance in the modern JupyterLab, and visit `src/cli-setup`.\

Either run every command one-by-one or use:
```bash
src/cli-setup/uv-path-activate.sh
```
Optionally export your AWS profile by:
```bash
src/cli-setup/export-profile.sh
```

...and that's all!

Now you have installed UV binaries, sync with and uv.lock file and activated the environment. We are going to use the custom CLI for our data pipeline.\
Visit `docs/data-pipeline.md` for more.