# OxIOD Dataset Download and Splitting Guide:

- Download the dataset from here: http://deepio.cs.ox.ac.uk/
- Within each folder in the dataset (in our case: ```handbag```, ```handheld```, ```pocket```, ```running```, ```slow_walking``` and ```trolley```), put files similar to the ```.txt``` files provided in this folder. They refer to which IMU files (and ground truth files to import). The ```.txt``` files we gave are just examples. You have to create splits through them for each folder.
- Check the ```data_utils.py``` file to see how data is imported.
