# Neural Networks in Fitting Data: Application to DEER Data
### Udacity DataScience ND: Capstone

## Project Definition
### Project Overview
* Train a convolutional neural network to detect oscillating features of experimental pulsed-EPR trace.
* To train the tested networks on simulated data.
### Problem Statement
* Is there a convolutional neural network (CNN) alternative to current fitting procedures.
* Vary the CNN components until adequate fitting is accomplished.
### Metrics
* For comparing the predicted curve to the target curve a mean-squared error (mse) loss function is used.
paraphrasing from https://pytorch.org/docs/stable/nn.html : mse = &#931; (target<sub>n</sub> - predicted<sub>n</sub>)<sup>2</sup> 
* The mean, median, and 95<sup>th</sup> percentile values of the mse are used to compare models.

## Analysis
* The use of simulated data alleviated issues that is presented by real data.
* Below are the statistics for four models tested.
<table>
   <tr> <td colspan<td colspan="9">Training/Validation Statistics</td>  </tr>
   <tr> <td>Model</td> <td colspan="2">1A</td> <td colspan="2">1B</td> <td colspan="2">2A</td> <td colspan="2">2B</td> </tr>
   <tr> <td></td> <td>training</td> <td>validation</td> <td>training</td> <td>validation</td> <td>training</td> <td>validation</td> <td>training</td> <td>validation</td> </tr>
   <tr> <td>mean</td> <td>1.79e-6</td> <td>1.62e-6</td> <td>1.89e-6</td> <td>1.73e-6</td> <td>1.74e-6</td> <td>1.52e-6</td> <td>9.58e-7</td> <td>1.27e-6</td> </tr>
   <tr> <td>median</td> <td>4.53e-7</td> <td>5.74e-7</td> <td>5.31e-6</td> <td>6.74e-7</td> <td>3.99e-7</td> <td>5.13e-7</td> <td>2.23e-7</td> <td>3.80e-7</td> </tr>
   <tr> <td>95<sup>th</sup> percentile</td> <td>8.54e-6</td> <td>6.90e-6</td> <td>8.80e-6</td> <td>7.02e-6</td> <td>8.43e-6</td> <td>6.62e-6</td> <td>4.26e-6</td> <td>5.70e-6</td> </tr>
   <tr> <td>minimum</td> <td>1.76e-8</td> <td>1.82e-8</td> <td>2.26e-8</td> <td>2.83e-8</td> <td>1.09e-8</td> <td>1.43e-8</td> <td>6.02e-9</td> <td>7.23e-9</td> </tr>
   <tr> <td>maximum</td> <td>6.21e-5</td> <td>5.10e-5</td> <td>6.23e-5</td> <td>3.42e-5</td> <td>6.37e-5</td> <td>3.52e-5</td> <td>6.01e-5</td> <td>8.76e-5</td> </tr>
</table>

* These are the plots for the data that is closest to the 95th percentile for model 1B. More figures are in the [write UP](./static/Deer_CNN.pdf) and in the Dashboard
![Plot of the 95th percentile for model 1B and how the rest of the models worked](https://github.com/RS-ND/Capstone-Project/blob/master/images/1B_95.png)

## Methodology
* The complete data trace with its various oscillations are considered one feature. This simulated data does not require any processing steps.

## Conclusion
### Reflection
*  Neural networks are used in in a variety of denoising applications, most notably in image analysis. The goal of this work was to take these principles of denoising data and to create a neural network for data obtained from pulsed electron paramagnetic resonance (pulsed-EPR) experiments. This denoised data could then be used in previously validated regression routines to determine the underlying distance distribution. Ultimately, the intermediate step of denoising data was omitted and the convolutional neural networks (CNN) were applied directly to determination of the distance distribution. While the results from this analysis are promising, additional work is needed to bring CNN to regular use in determining distance distributions from pulsed-EPR data.
* One interesting aspect of this project was further exploring convolution and getting a better understanding of padding and the ability to apply the kernel spread across the input signal.
* One of the most difficult aspects was carrying this out without a gpu.

### Improvement
* Simply go back to the original concept and find a CNN that is robust and reliable at denoising the noisy input Deer traces.
* Explore the use of batch normalization and dropout layers.
* Increase the complexity of the fully-connected layer compared to model 2B.
* Use a larger and more complex training set that includes non-Gaussian peaks in the distributions.
* Preprocessing of the Deer traces into spectrograms that can then be used as input with pretrained image CNNs, such as ResNet, AlexNet, or vgg.
* Take the predicted distance distribution and calculate the associated Deer trace and add an additional loss function that compares this predicted Deer trace to the input Deer trace. This could improve the end result as the actual shape of the distance distribution is not required, but the best shape that fits the input Deer trace is.

[Write UP](./static/Deer_CNN.pdf) (also available using the Dashboard)


Dashboard/Python Files (**__WebApp__**)
> __Deer_CNN.py: run at command line and in a web browser open "localhost:3001"__

* Requires both the templates and static folders to be in the directory containing Deer_CNN.py

* Requires data hosted at: https://drive.google.com/open?id=1twFOE14AvB7hkQ6_hxlw2PNr8zUtcF3g (slow download)
  * Both the models and data directories should be placed in the folder where Deer_CNN.ipynb is located
    * Both the training and validation data needs to be unzipped in folders named train and valid, respectively.

This Dashboard:
* __Has an overview of the project__
* __Allows for exploration of the relationship between a distance distribution and a deer trace__
* __Several explorations of the results of the CNNs__


Jupyter Notebook (ipynb) 
> __Deer_CNN.ipynb__

* Requires models_load.py


HTML file
> __Deer_CNN.html__

    export of the above Jupyter notebook

 
### Note: The Jupyter notebook requires files not located on GitHub (see link above)
     
     

