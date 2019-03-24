# Neural Networks in Fitting Data: Application to DEER Data
### Udacity DataScience ND: Capstone


Summary

>  Neural networks are used in in a variety of denoising applications, most notably in image analysis. The goal of this work was to take these principles of denoising data and to create a neural network for data obtained from pulsed electron paramagnetic resonance (pulsed-EPR) experiments. This denoised data could then be used in previously validated regression routines to determine the underlying distance distribution. Ultimately, the intermediate step of denoising data was omitted and the convolutional neural networks (CNN) were applied directly to determination of the distance distribution. While the results from this analysis are promising, additional work is needed to bring CNN to regular use in determining distance distributions from pulsed-EPR data.

[Write UP](./static/Deer_CNN.pdf) (also available using the Dashboard)


Dashboard/Python Files 
> __Deer_CNN.py: run at command line and in a web browser open "localhost:3001"__

    This Dashboard:
        Has an overview of the project
        Allows for exploration of the relationship between a distance distribution and a deer trace
        Several explorations of the results of the CNNs.
          
* Requires both the templates and static folders to be in the directory containing Deer_CNN.py
          

Jupyter Notebook (ipynb) 
> __Deer_CNN.ipynb__

* Requires data hosted at: https://drive.google.com/open?id=1twFOE14AvB7hkQ6_hxlw2PNr8zUtcF3g
  * Both the models and data directories should be placed in the folder where Deer_CNN.ipynb is located
    * The both the training and validation data needs to be unzipped.


HTML file
> __Deer_CNN.html__

    export of the above Jupyter notebook

 
### Note: The Jupyter notebook requires files not located on GitHub (see link above)
     
     

