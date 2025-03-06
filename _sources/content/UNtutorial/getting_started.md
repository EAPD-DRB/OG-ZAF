---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(Chap_UNtutor_getstart)=
# Getting Started

(Sec_UNtutor_schedule)=
## Schedule
For the March 3-7, 2025 United Nations `OG-ZAF` training in Cape Town, we will be following the schedule in {numref}`table-schedule`.

```{list-table} UN OG-ZAF 5-day training schedule
:header-rows: 1
:name: table-schedule

* - Day
  - Session
  - Topic
  - Materials
* - Mon.
  - Morning
  - Organizer introductions <br> [Setup Python](https://eapd-drb.github.io/OG-ZAF/content/UNtutorial/getting_started.html#install-python), [Git, GitHub](https://eapd-drb.github.io/OG-ZAF/content/UNtutorial/getting_started.html#installing-git-and-github), and [OG-ZAF](https://eapd-drb.github.io/OG-ZAF/content/UNtutorial/getting_started.html#fork-and-clone-og-zaf-repository)
  - [Intro slides](https://github.com/EAPD-DRB/OG-ZAF/blob/main/docs/book/content/UNtutorial/materials/OG-ZAF-Open.pdf)
* -
  - Afternoon
  - Theory: ["Simple" 3-period-lived agent model](https://eapd-drb.github.io/OG-ZAF/content/UNtutorial/3perOG.html)
  - [Solutions](https://github.com/EAPD-DRB/OG-ZAF/blob/main/docs/book/content/UNtutorial/solutions/3perOG/)
* - Tue.
  - Morning
  - Review 3-period-lived-agent exercises (solutions in [this folder](https://github.com/EAPD-DRB/OG-ZAF/blob/main/docs/book/content/UNtutorial/solutions/3perOG/)) <br> Review OG-Core and OG-ZAF modules <br> Quick Git and GitHub workflow review
  -
* -
  - Afternoon
  - Running OG-ZAF, inputs, outputs <br> <br> Calibrating OG-ZAF, current state, still to do
  - [I/O slides](https://github.com/EAPD-DRB/OG-ZAF/blob/main/docs/book/content/UNtutorial/materials/OG-ZAF-inputoutput.pdf) <br> [I/O Colab notebook](https://colab.research.google.com/drive/10toq2-SIctowK-yTL-RBsLMiQQGJX4QW?usp=sharing) <br> [Calibrate slides](https://github.com/EAPD-DRB/OG-ZAF/blob/main/docs/book/content/UNtutorial/materials/OG-ZAF-CurrentState.pdf)
* - Wed.
  - Morning
  - Running OG-ZAF: Revisit some reforms from 2-day visit <br> Talk about new reforms <br> Create project teams (see "[Research Projects](https://eapd-drb.github.io/OG-ZAF/content/UNtutorial/projects.html)" chapter)
  - [Reforms slides](https://github.com/EAPD-DRB/OG-ZAF/blob/main/docs/book/content/UNtutorial/materials/OG-ZAF-PrevAndNewReforms.pdf) <br> [Notebooks](https://github.com/EAPD-DRB/OG-ZAF/blob/main/docs/book/content/UNtutorial/materials/PrevReformsNotebooks/notebooks) <br> [Run scripts](https://github.com/EAPD-DRB/OG-ZAF/blob/main/docs/book/content/UNtutorial/materials/PrevReformsNotebooks/run_scripts)
* -
  - Afternoon
  - OG-ZAF output: Tools to visualize/tabulate output <br> OG-ZAF built-in calibration helps
  - [Built-in tools notebook](https://github.com/EAPD-DRB/OG-ZAF/blob/main/docs/book/content/UNtutorial/materials/OG-ZAF_builtintools.ipynb)
* - Thu.
  - Morning
  - Calibrating OG-ZAF: Issues and hot spots
  -
* -
  - Afternoon
  - Calibrating OG-ZAF: Issues and hot spots
  -
* - Fri.
  - Morning
  - Open work, project hackathon, office hours <br> Advanced topics: Adding trade, connecting to other models
  -
* -
  - Afternoon
  - Presentation of projects <br> Future work, research, collaboration, final topics <br> Closing remarks
  -
```

(Sec_UNtutor_python)=
## Install Python
`OG-ZAF` is a large-scale overlapping generations macroeconomic model of Philippine fiscal policy. It is written in the [Python](https://www.python.org/) programming language. You will need to have some distribution of Python loaded on your computer to run the code. We recommend installing the [Anaconda](https://www.anaconda.com/download) distribution of Python. This is the most widely used Python distribution. And it has package management features, like conda environments, that we will make use of.

### Verifying you have already installed Python and Conda
If you have already installed the Anaconda distribution of Python, do the following steps to verify your installation.

#### For Windows and Mac
If you are using computer with a Windows operating system, open your Anaconda prompt. If you are using a Mac operating system, open your terminal.
- To see if you have Python installed, type `python --version`. This command should result in output like `Python 3.12.7`.
```{code}
>>> python --version
Python 3.12.7
```
- To see if you have Anaconda's conda package installed type `conda --version`. This command should results in output like `conda 24.9.2`.
```{code}
>>> conda --version
conda 24.9.2
```

#### For Linux
If you are using a computer with a Linux operating system, open your terminal.
- To see if you have Python installed, type `python -version`. This command should result in output like `Python 3.12.7`.
```{code}
>>> python -version
Python 3.12.7
```
- To see if you have Anaconda's conda package installed type `conda -version`. This command should results in output like `conda 24.9.2`.
```{code}
>>> conda -version
conda 24.9.2
```

### Installing Anaconda distribution of Python
To install the Anaconda distribution of Python, go to https://www.anaconda.com/download and select "Skip registration" to avoid giving your email address.


## Installing Git and GitHub

### Verifying you have already installed Git

#### On Mac or Windows
Open your terminal (Mac) or command prompt (Windows) and type `git --version`. You should get output like `git version 2.37.2`.
```{code}
>>> git --version
git version 2.37.2
```

#### On Linux
Open your terminal and type `git -version`. You should get output like `git version 2.37.2.
```{code}
>>> git -version
git version 2.37.2
```

### Installing Git
If you do not already have Git installed on your computer, we recommend that you follow the instructions on the GitHub page https://github.com/git-guides/install-git for installing Git. This page has downloadable executable installers that are easy to use.


### Basic configuring of Git on your machine
Once you have Git installed, you will need to configure some of the basic settings in Git. To view all of your Git settings, you can type the following into your computers terminal:
```{code}
git config --list --show-origin
```
When getting set up, it’s important to enter your credentials so that `git` on your local machine is linked to your account on GitHub. You’ll do this by first entering your name:
```{code}
git config --global user.name "Your Name"
```
Then, you’ll enter your email (using the email that you used to register your account on GitHub.com):
```{code}
git config --global user.email yourname@example.com
```
You can also set your default text editor for use with `git` by following the example below, which makes vim the default:
```{code}
git config --global core.editor vim
```

For more information on configuring `git`, see the full instructions from `git` [here](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup).


### Set up a GitHub account
You will need a GitHub account to properly interact with OG-ZAF. This will allow you to interact with the repository with a wide range of collaborative functionalities, including forking repositories, creating issues and discussions, and submitting pull requests. To set up a GitHub account, follow [these instructions](https://docs.github.com/en/get-started/start-your-journey/creating-an-account-on-github) at GitHub.com.

Most likely, the free organization account will be the right place to start for you. We recommend choosing a username suitable for a professional setting, as this will be your public profile on GitHub.

## Fork and clone OG-ZAF repository
1. Forking the OG-ZAF repository means that you are making a copy of that repository on your GitHub account in the cloud.
    - Go to the UN GitHub organization's main repository for OG-ZAF (https://github.com/EAPD-DRB/OG-ZAF).
    - In the upper-right area of the browser page, click the "Fork" button and select "Create fork". This will create an exact copy of the OG-ZAF repository on your account. When you do this, you should see that the URL to the page has changed to `https://github.com/[YourGitHubHandle]/OG-ZAF`.

2. The next step is to clone the repository from its current place in the cloud to your local computer's hard drive.
    - Open your terminal or Anaconda prompt
    - Navigate to the folder where you want this repository to reside. Make sure this is not a location on your hard drive that is mapped from the cloud. This file should live on your local computer. You already have the repository in the cloud on your GitHub account.
    - Copy the contents of your repository in the cloud to your hard drive by typing: `git clone https://github.com/[YourGitHubHandle]/OG-ZAF.git`
    - Change directory to this new directory by typing: `cd OG-ZAF`
    - Create an additional git remote named "upstream" that points to the main UN remote repository by typing: `git remote add upstream https://github.com/EAPD-DRB/OG-ZAF.git`

```{figure} ./images/GitFlowDiag.png
---
scale: 50%
align: center
name: GitFlowDiag
---
Flow diagram of Git and GitHub workflow
```

## Create the ogzaf-dev conda environment
[Conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) are a functionality that comes with the Anaconda distribution of Python. Conda environments allow the users across operating system platforms and different hardware configurations to run Python code in an environment that has the same packages, functionality, and results. A Conda environment is similar to a Docker image.

If you have installed the Anaconda distribution of Python and you have cloned your OG-ZAF fork of the repository to your local machine, you can create the `ogzaf-dev` conda environment by doing the following steps:
- Open your terminal or command prompt and navigate to the OG-ZAF repository folder on your hard drive.
- Type the following command: `conda env create -f environment.yml`
- Once the environment has been created, you must activate it: `conda activate ogzaf-dev`
- In the activated `ogzaf-dev` conda environment, install the `ogzaf` Python package directly from your repository by typing: `pip install -e .`

Now you have the `ogzaf` Python package installed in your `ogzaf-dev` conda environment. Now you will be able to run the modules of the OG-ZAF model from scripts and from Jupyter notebooks.

## Using Jupyter notebooks
A nice way to execute lines of code on your local computer is to use Jupyter notebooks. The `jupyter` package is installed as part of the `ogzaf-dev` conda environment from the previous step. You can open a Jupyter notebook directly in VS Code, or you can open one from your terminal or command prompt.

### Open Jupyter notebook from terminal or command prompt
- If you are using Mac or Linux, open your terminal. If you are using Windows, open your command prompt.
- Navigate to the folder of the OG-ZAF repository on your local machine.
- Activate the `ogzaf-dev` conda environment by typing `conda activate ogzaf-dev`.
- Open a Jupyter notebook session by typing `jupyter notebook`. This will open a local server page that opens in your browser. This page will show the directory where you are currently working.
- Either click the "New" button in the upper-right portion of the screen, or select "File" then "New" then "Notebook" from the menu at the upper-right. Make sure to select the `ogzaf-dev` kernel.

Once you have completed these steps, you can interactively write code and execute it in steps using the Python code cells in the Jupyter notebook. You can also write text descriptions in the markdown cells.

## Choosing a text editor
Using a good text editor for your coding is a key productivity choice. We recommend the [VS Code (Visual Studio Code)](https://code.visualstudio.com/) editor from Microsoft (download from https://code.visualstudio.com/Download). This text editor is free, it is open source, and it has the largest community of active users and active developers. It also has a ton of extensions that help you customize and increase the efficiency of your coding workflow.
