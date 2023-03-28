This is a demo accompanying the article, "Basket Analysis with Computer Vision", Eugene Asahara - https://vmblog.com/archive/2023/03/17/basket-analysis-with-computer-vision.aspx#.ZCL8IrLMJD9 

Setup
=====

This demo is conducted through an Anaconda/Jupyter notebook: Computer_Vision_Demo.ipynb

1. Install Anaconda.
2. Set up an Azure Computer Vision resource.
   2a. Retrieve a key and the endpoint for the Azure Computer Vision Resource.
3. Download the samples repository into c:\samples-main. 
   3a. Be sure the c:\samples-main\BasketAnalysis files are there.
4. Ensure the Python packages in the requirements.txt file are installed.
5. Edit the following keys in the .env file:
   5a. COMPUTER_VISION_KEY="Your Azure Computer Vision Key"
   5b. COMPUTER_VISION_ENDPOINT="Your Azure Computer Vision Endpoint"
   5c. Directory where images we wish to crack are stored: COMPUTER_VISION_IMAGE_PATH="C:/samples-main/BasketAnalysis/BasketImages/"
   5d. Directory where the basket analysis data files are creates: COMPUTER_VISION_SAVE_DATA_PATH="any existing directory on your machine"
6. Open an Anaconda prompt.
   6a. Enter the command: jupyter lab --notebook-dir=c:\samples-main\BasketAnalysis

Please contact me at eugene.asahara@kyvos.io if you have any questions.

Notes:
======
I had a problem with the .env file. In the Anaconda notebook, I got the error, "Python-dotenv could not parse statement starting at line 3". I removed quotes and it seemed to work.
