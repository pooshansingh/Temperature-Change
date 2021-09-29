#  Temperature Change
Data Analysis, Visualization and Prediction of global temperature change using FAOSTAT Data.

![Image](https://images.unsplash.com/photo-1584701782188-b44dc2815522?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=968&q=80)

## Introduction and Purpose

Little has to be said when the global populus is experiencing climate change firsthand. But relations have to be established between organic data and the visible change if we must understand it better. Except for strengthening the agruments in favor of working against climate change, it also ensures we use the right tools to do so. Using UN's Global Tempperature change data. Which we first clean and restructure it to our ease. The aim is to understand in depth, the patterns and relationships in temperature rise over time. Next logical thing to do is co-relate it with rising CO-2 emissions, loss of forest cover, Air Quality index as factors of cause. Linear and Polynomial Regression is used to train our data and get temperature rise prediction till year 2059. The forecast is done with 2 and 5 degrees of polynomials.   

Prelimnary analysis of the data realised more regular surges in temperature post 1978. We also perform a case study on patterns in temperature change for Antarctica.   

<!--The project title should be self explanotory and try not to make it a mouthful. (Although exceptions exist- **awesome-readme-writing-guide-for-open-source-projects** - would have been a cool name)

Add a cover/banner image for your README. **Why?** Because it easily **grabs people's attention** and it **looks cool**(*duh!obviously!*).

The best dimensions for the banner is **1280x650px**. You could also use this for social preview of your repo.

There are endless badges that you could use in your projects. And they do depend on the project. Some of the ones that I commonly use in every projects are given below. 

I use [**Shields IO**](https://shields.io/) for making badges. It is a simple and easy to use tool that you can use for almost all your badge cravings. -->

<!-- Some badges that you could use -->

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/navendu-pottekkat/awesome-readme?include_prereleases)

![GitHub commit activity](https://img.shields.io/github/commit-activity/m/pooshansingh/Temperature-Change?style=plastic)

![GitHub all releases](https://img.shields.io/github/downloads/pooshansingh/Temperature-Change/total?style=plastic)

![GitHub issues](https://img.shields.io/github/issues-raw/pooshansingh/Temperature-Change?style=plastic)

![GitHub](https://img.shields.io/github/license/pooshansingh/Temperature-Change?style=plastic)


![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/pooshansingh/Temperature-Change)

# Table of contents

- [Demo-Preview](#demo-preview)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Contribute](#contribute)
    - [Sponsor](#sponsor)
    - [Adding new features or fixing bugs](#adding-new-features-or-fixing-bugs)
- [License](#license)
- [Footer](#footer)


# Demo-Preview

The model is quite robust in terms of consistency and also efficient with formulated results. 

A Case Study on Antarctica was performed prior to looking for insights over the larger dataset in an attempt to spot patterns of temperature change.

![Image](https://drive.google.com/uc?export=view&id=1i9956IxiKa6rsmQBwK3DXF-9vOz7DM3Y)

#### Yearly magnitude of temperature change in Antarctica

![Image](https://drive.google.com/uc?export=view&id=11_hUJlWAj6wJYHr4iPqbwwCT0_n1wtAS)

#### Mean Monthly change (grouped and individual)

![Image](https://drive.google.com/uc?export=view&id=1hbpDvkBEAY9OAJwRJ1rackt5UJD6uGw5) 

#### Standard Deviation in temperature change in Antarctica

![Image](https://drive.google.com/uc?export=view&id=1yHrthLmZzMvkyY8Y_ZdjNfRPjriwtDMg) 

#### Plotted annual temperature rise 

<!-- Add a demo for your project -->
The learnings we get from studying prelimnary results are utilised to devise more detailed plans for further data wrangling, analysis, and visualisations. 

We use these results to try and first find the global average temperature change so we can visually second the increasing trends earlier noticed.

![Image](https://drive.google.com/uc?export=view&id=1vKKuVxfPr94vRwW8sSvKD2JGAtb_WUhh) 

Linear and Polynomial Regression models were trained with the dataset to realise a prediction of temperature change in forseeable future. Final inference indicated the global average above 4°C, unsurprisingly higher than the desired 1.5°C mark.

![Image](https://drive.google.com/uc?export=view&id=1cmEJzip366x_Qp-DLvr_zfas1n1_YbMB) 

# Installation
[(Back to top)](#table-of-contents)

<!-- *You might have noticed the **Back to top** button(if not, please notice, it's right there!). This is a good idea because it makes your README **easy to navigate.*** 

The first one should be how to install(how to generally use your project or set-up for editing in their machine).

This should give the users a concrete idea with instructions on how they can use your project repo with all the steps.

Following this steps, **they should be able to run this in their device.**

A method I use is after completing the README, I go through the instructions from scratch and check if it is working. -->

To use this project, first clone the repo on your device using the command below:

```git init```

```git clone https://github.com/pooshansingh/Temperature-Change.git```

Or open the .py extension file in your editor and make sure it has needed modules to execute the code. Remember that once we delete one Austria Dataframe as it consists of repeated NaN values from its preceeding set, it has to be hashed out in order to not delete the actual row.

![Image](https://drive.google.com/uc?export=view&id=1G5iKHXXdM8fqGSAPg8CDjxtnkt6D2sYK) 

Model is now ready to run. 

# Development
[(Back to top)](#table-of-contents)

Data can be grouped to the field as per your modifcation choice. Our intention to have them by country is to gather region specific results in any timeline. 

A Dataframe 'temperature_change' is created where we locate the countries and uniquely identify within them, monthly temperature change values over the years. Same is done for our Case Study on Antarctica. An Seaborn lineplot vivdly describes the upper shift of magnitude in the continent's temperature. Standard Deviation plot for the year 2010 also shows a comparative loss of winter owing to the longer durations now needed for the ice to form. 

![Image](https://drive.google.com/uc?export=view&id=1BaOazl1G9appzC9srv9Xwozx2qt3IWPb) 

### Global temperature change, Data Wrangling and Analysis

A similar pattern is obsevered over the global dataset, all each discreet regions of their own. Providing a streamlined understanding of patterns in climate change through grouped countries. Further wrangling is performed on the larger dataset using our insights from the case study. 

![Image](https://drive.google.com/uc?export=view&id=1-cRyRkI0MxLJsUMfDLph4jsCO3OY7T69) 

Global average is found using df 'temperature_change', elements of whose are grouped in years. The mean hence found is saved in dataframe 'average_temp', similar logic is used to find the mean by country.
![Image](https://drive.google.com/uc?export=view&id=1fVIjvHn-h5SERN9E6Muf3kYewyCHKcoN)

### Fitting model into training data and generating predictions

Simple Linear Progression is used to fit and train our model. 
![Image](https://drive.google.com/uc?export=view&id=1wX0zzxlrasXKGrojgktdpJyApHFZsrwD)  

Prediction points are randomly generated and sorted by values as strings in order to get the test data in primary dataframe, giving us the final results. The RMSE score reads to be 0.9721. 

![Image](https://drive.google.com/uc?export=view&id=1gDCuIyxoWjGDWTFZITQXI_UWpiuzKTL9)
Predicted and actual value comparison density plot. We use the same logic to test the trained data under Polynomial Regression. The obtained values are visualised in the grouped graphs showing both the compiled temperature changed data, linear as well as degreed regressions. 


<!-- This is the place where you give instructions to developers on how to modify the code.
You could give **instructions in depth** of **how the code works** and how everything is put together.

You could also give specific instructions to how they can setup their development environment.

Ideally, you should keep the README simple. If you need to add more complex explanations, use a wiki. Check out [this wiki](https://github.com/navendu-pottekkat/nsfw-filter/wiki) for inspiration. -->



# Contribute
[(Back to top)](#table-of-contents)

<!-- This is where you can let people know how they can **contribute** to your project. Some of the ways are given below.

Also this shows how you can add subsections within a section. -->

### Adding new features or fixing bugs

Pull request or issue submission widgets available on the top.


<!-- This is to give people an idea how they can raise issues or feature requests in your projects. 

You could also give guidelines for submitting and issue or a pull request to your project.

Personally and by standard, you should use a [issue template](https://github.com/navendu-pottekkat/nsfw-filter/blob/master/ISSUE_TEMPLATE.md) and a [pull request template](https://github.com/navendu-pottekkat/nsfw-filter/blob/master/PULL_REQ_TEMPLATE.md)(click for examples) so that when a user opens a new issue they could easily format it as per your project guidelines.

You could also add contact details for people to get in touch with you regarding your project. -->

# License

[GNU General Public License version 3](https://opensource.org/licenses/GPL-3.0)
[(Back to top)](#table-of-contents)


<!-- Adding the license to README is a good practice so that people can easily refer to it.

Make sure you have added a LICENSE file in your project folder. **Shortcut:** Click add new file in your root of your repo in GitHub > Set file name to LICENSE > GitHub shows LICENSE templates > Choose the one that best suits your project!

I personally add the name of the license and provide a link to it like below. -->


# Footer
[(Back to top)](#table-of-contents)

<!-- Let's also add a footer because I love footers and also you **can** use this to convey important info.

Let's make it an image because by now you have realised that multimedia in images == cool(*please notice the subtle programming joke). -->
<!-- Add the footer here -->

<!-- ![Footer](https://github.com/navendu-pottekkat/awesome-readme/blob/master/fooooooter.png) -->
