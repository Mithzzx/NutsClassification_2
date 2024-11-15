# Bone Cancer Prediction

## Description
This is colaborative team project of 4 members, where we were tasked with building 2 prediction model using different algorithms and compare both the algorithms based on some metric (e.g. accuracy, precision, etc), then display the comparision in the form of graph. 

We should then test the model in:
1) Jenkins
2) Docker
3) Selenium


### Our Approach
We used *Linear Regression* and *Random Forest Regression* as the algorithms to build this model. We chose R², MAE (Mean Absolute Error) and MSE (Mean Squared Error) as the metrics for prediction, and used **matplotlib** library for plotting comparison graph of the 2 algorithms.

Then, displayed the results using flask.

## Steps

### Building Files
1) train_model.py: which has regression model code
2) requirements.txt: which has the required library names
3) AEP_hourly.csv: .csv dataset which includes the data regarding counsumer forcast prediction
4) Dockerfile: which contains the commands to be run in docker

### Git & Other Commands used in Command Prompt
1) `mkdir <file_name>` to create the folder
2) `cd <file_name>` to go inside the folder
3) `git init`, initializing folder as git repository
4) manually add all the files
5) `git add .` to add all the files into git track
6) `git commit -m "<message>"`, it commits whatever is there in the track
7) `git remote add origin <GitHub repository link>`, builds connection between git and github
8) `git push --set-upstream origin master`, sends all the files from git to github.


### Jenkins
1) created a new item with item name as **<item_name>**
2) selected *Freestyle project* as the item type
3) selected *Git* in **Source Code Management**
4) added *Execute Windows batch command* step in **Build Steps**
5) Saved the Configurations
6) Clicked build now

#### Output
  ![WhatsApp Image 2024-11-15 at 21 28 00_9a633f03](https://github.com/user-attachments/assets/2dee6897-2297-4f64-9bb3-a248522ce57c)




### Dockers
1) used `cd` to go forward and `cd..` to do backward, to get to the correct path in command prompt (i.e. the folder where all the files including docerfile is located.
2) executed the command `docker build -t <docker_image_name> .` to build the docker image.
3) executed the command `docker run <docker_image_name>` to run the docker image.

#### Output
  ##### Command Prompt
  ![test 2 1](https://github.com/user-attachments/assets/b115cf29-e080-40f9-88d3-fd36c9646be1)

  ##### Docker Hub
  ![test 2 2](https://github.com/user-attachments/assets/5f6e5a2d-f752-496f-b4a3-99c874512635)



### Selenium
1) integrated flask into app.py. And build an html file called index.html inside templates folder.
2) created selenium_test.py file which contains the selenium code for autotesting app.py

#### Output
  ##### Flask app (Command Prompt & Web Browser)
  ![test 2 1](https://github.com/user-attachments/assets/11cf6710-ec3b-4e9a-a613-21755a74dee3)

  ##### Selenium (Command Prompt)

  ![test 2 2](https://github.com/user-attachments/assets/b4c37d8c-bb8c-4f24-af09-5a6696e7de7d)
