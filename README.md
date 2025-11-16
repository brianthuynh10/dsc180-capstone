# DSC180 - Quarter 1 Project
## Setup
### Step 0: Prerequisites
<ul>
<li>
Make sure you have Docker Desktop install on your computer (if not, install here for on [Mac](https://docs.docker.com/desktop/setup/install/mac-install/) or 
[Windows](https://docs.docker.com/desktop/setup/install/mac-install/](https://docs.docker.com/desktop/setup/install/windows-install/ )
</li> 
<li>
Create a Weights and Biases account ([here](https://wandb.ai/site/)) 
  <ol type="1"> 
    <li> Once you reach the home page, navigate to the top right corner of your profile and click on it</li>
    <li> In the drop down, click on API key where you will be directed to page where you can copy your API key </li>
    <li> Copy the key and place it somewhere you'll remember! You can always navigate back to this page if you lose it</li>
  </ol>
</li>
</ul>

### Step 1: Cloning the repository
Open and paste the following command
```
git clone https://github.com/brianthuynh10/dsc180-capstone-recreate.git
```

### Step 2: Docker Image Setup: 
Make sure you have Docker open, then navigate to the repository then use the following command in your terminal, 
```
docker buildx build -platform [TARGET_OS]/[TARGET_CPU] -t [DOCKER_USERNAME]/[IMAGE]:[TAG] [CONTEXT_PATH]
```
Example: <br>
<b> NOTE: </b> Since DSMLP does not run using Mac and our data resides on the cluster, we suggest using `linux/amd64` for your ```[TARGET_OS]/[TARGET_CPU]```
```
docker buildx build - platform linux/amd64 -t brianthuynh10/dsc-env:latest .
```
Next, push your image using the following command,
```
docker push [DOCKER_USERNAME]/[IMAGE]:[TAG]
```
Example: 
```
docker push brianthuynh10/dsc-env:latest
```
<b> Make sure your image is public, otherwise DSMLP cannot pull your image! You can change this seting by logging into DockerHub on your browser and changing your image's settings </b> 

### Step 3: DSMLP Cluster: 
First you need to SSH into the DSMLP, then once you're in the jumpbox server run the follwing command to ensure you're using the Docker image you created earlier. Note: You can add the tag `-b` if you want to create a background pod because the model will take a while to train
```
launch.sh \
    -W DSC180A_FA25_A00 -G b1100018875 \
    -i [DOCKER_USERNAME]/[IMAGE]:[TAG] \
    -c [NUMBER_OF_CPUs] -m [SIZE_OF_RAM] -g [NUMBER_OF_GPUs] -v [GPU_VARIANT] \
    -P Always -T -s
```
Example: The configuration below is sufficient to run the model smoothly. Just make sure to have a GPU! You can check whichever one is free [here](https://datahub.ucsd.edu/hub/status)
```
launch.sh \
    -W DSC180A_FA25_A00 -G b1100018875 \
    -i brianthuynh10/dsc-env:latest \
    -c 8 -m 32 -g 1 -v 2080ti \
    -P Always -T -s \
```
### Step 4: Copying the Repo to DSMLP (if you already did that, skip to step 5)
If you cloned the repo to your local machine, you'll have to SCP the repo to DSMLP. First we'll need to zip the entire repository using the command: 
```
zip -r [ZIP_FOLDER_NAME].zip dsc180-capstone-recreate
```
Then use the following command to send it to DSMLP
```
scp <FILE_TO_TRANSFER> [USERNAME]@dsmlp-login.ucsd.edu:<FILE_NAME_FOR_DSMLP>
```
You should see the file appear in your private folder in DSMLP where you can go into the DSMLP terminal to unzip using
```
unzip [ZIP_FOLDER_NAME].ZIP
```
### Step 5: Running the Script
Inside DSMLP, navigate to path of the repo (if you didn't change any names the directory should be `private/dsc180-capstone-recreate`). From there you can run the command,
```
python3 main.py all > log.txt &
```
At some point early in the run, you'll be prompted to enter your W&B API key which will allow you to see training status and the results on the test set. There will also be a `log.txt` file training progress and an output folder the saves the best model incase the process crashes during training.  Once the process is over, you'll be able to see graphs of model's predicitons on the test set and at each step on the validation set. 








