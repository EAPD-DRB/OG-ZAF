(Chap_ContribGuide)=
# Contributor Guide

The purpose of this guide is to provide the necessary background such that you can make improvements to the `OG-ZAF` and share them with others working on the model.

`OG-ZAF` code is tracked by using [`Git`](https://help.github.com/articles/github-glossary/#git) version control software along with the [GitHub.com](https://github.com/) web platform for `Git` workflow and collaboration. Following the next steps will get you up and running and contributing to the model even if you've never used `Git` or other version control software.

If you have already completed the {ref}`Sec_SetupPython` and {ref}`Sec_SetupGit` sections, you can skip to the {ref}`Sec_Workflow` section.


(Sec_SetupPython)=
## Setup Python

`OG-ZAF` is written in the Python programming language. Download and install the free and most recent Anaconda distribution of Python and associated libraries from [Anaconda.com](https://www.anaconda.com/products/individual#Downloads).[^recent_python] You should do this even if you already have Python installed on your computer because the Anaconda distribution contains additional Python packages that are used by `OG-ZAF` (many of which are not included in other Python installations). You can install the Anaconda distribution without having administrative privileges on your computer and the Anaconda distribution will not interfere with any Python installation that came as part of your computer's operating system.


(Sec_SetupGit)=
## Setup Git

1. Create a [GitHub](https://github.com/) user account.

2. Install Git on your local machine by following steps 1-4 on [Git setup](https://help.github.com/articles/set-up-git/).

3. Tell Git to remember your GitHub password by following steps 1-4 on [password setup](https://help.github.com/articles/caching-your-github-password-in-git/).

4. Sign in to GitHub and create your own [remote](https://help.github.com/articles/github-glossary/#remote) [repository](https://help.github.com/articles/github-glossary/#repository) (repo) of `OG-ZAF` by clicking [Fork](https://help.github.com/articles/github-glossary/#fork) in the upper right corner of the [OG-ZAF GitHub repository page](https://github.com/EAPD-DRB/OG-ZAF). Select your username when asked "Where should we fork this repository?"

5. From your command line, navigate to the directory on your computer where you would like your local repo to live.

6. Create a local repo by entering at the command line the text, shown in the code blocks below after the `$` symbol.[^commandline_note] This step creates a directory called `OG-ZAF` in the directory that you specified in the prior step:

    ```
      $ git clone https://github.com/[github-username]/OG-ZAF.git
    ```

7. From your command line or terminal, navigate to your local `OG-ZAF` directory.

8. Make it easier to [push](https://help.github.com/articles/github-glossary/#pull) your local work to others and [pull](https://help.github.com/articles/github-glossary/#pull) others' work to your local machine by entering at the command line:

    ```
      $ cd OG-ZAF
      OG-ZAF$ git remote add upstream https://github.com/EAPD-DRB/OG-ZAF.git
    ```

9. Create a conda environment with all of the necessary packages to execute the source code.
The process of creating the `ogzaf-dev` conda environment can take up to 10 minutes. The pip install of the `OG-Core` dependency from [pypi.org](https://pypi.org/project/ogcore/).

    ```
      OG-ZAF$ conda env create
    ```

10. The prior command will create a conda environment called `ogzaf-dev`.
    Activate this environment as follows:

    ```
      OG-ZAF$ conda activate ogzaf-dev
    ```

11. Once the environment is activated, install the `ogzaf` package in the `ogzaf-dev` environment with all its modules by executing the following `pip install` command.

    ```
      (ogzaf-dev) OG-ZAF$ pip install -e .
    ```

If you have made it this far, you've successfully made a remote copy (a
fork) of the central `OG-ZAF` repo. That remote repo is hosted on GitHub.com at [https://github.com/EAPD-DRB/OG-ZAF](https://github.com/EAPD-DRB/OG-ZAF). You have also created a local repo (a [clone](https://help.github.com/articles/github-glossary/#clone)) that lives on your machine and only you can see; you will make your changes to
the OG-ZAF model by editing the files in the `OG-ZAF`
directory on your machine and then submitting those changes to your
local repo. As a new contributor, you will push your changes from your
local repo to your remote repo when you're ready to share that work
with the team.

Don't be alarmed if the above paragraphs are confusing. The following
section introduces some standard Git practices and guides you through
the contribution process.


(Sec_Workflow)=
## Workflow

(Sec_GitHubIssue)=
### Submitting a GitHub Issue

GitHub "issues" are an excellent way to ask questions, include code examples, and tag specific GitHub users.


(Sec_GitHubPR)=
### Submitting a GitHub Pull Request

The following text describes a typical workflow for changing
`OG-ZAF`.  Different workflows may be necessary in some
situations, in which case other contributors are here to help.

1. Before you edit the `OG-ZAF` source code on your machine,
   make sure you have the latest version of the central OG-USA
   repository by executing the following **four** Git commands:

   a. Tell Git to switch to the master branch in your local repo.
      Navigate to your local `OG-ZAF` directory and enter the
      following text at the command line:

    ```
        OG-ZAF$ git checkout master
    ```

   b. Download all of the content from the central `OG-ZAF` repo:
    ```
        OG-ZAF$ git fetch upstream
    ```
   c. Update your local master branch to contain the latest content of
      the central master branch using [merge](https://help.github.com/articles/github-glossary/#merge). This step ensures that
      you are working with the latest version of OG-ZAF:
    ```
        OG-ZAF$ git merge upstream/master
    ```
   d. Push the updated master branch in your local repo to your GitHub repo:
    ```
        OG-ZAF$ git push origin master
    ```
2. Create a new [branch](https://help.github.com/articles/github-glossary/#branch) on your local machine. Think of your branches as a way to organize your projects. If you want to work on this documentation, for example, create a separate branch for that work. If you want to change an element of the OG-ZAF model, create a different branch for that project:
    ```
     OG-ZAF$ git checkout -b [new-branch-name]
    ```
3. As you make changes, frequently check that your changes do not
   introduce bugs or degrade the accuracy of the `OG-ZAF`. To do
   this, run the following command from the command line from inside
   the `/OG-ZAF/ogzaf/` directory:
    ```
     OG-ZAF/ogzaf$  pytest -m "not needs_puf and not regression"
    ```
   Note that running this full suite of tests may take close to 24 hours (depending on your hardware).  To run the small set of tests that run on each pull request (and take about 40 minutes), use  `pytest -m "not needs_puf and not local and not regression"`.  If the tests do not pass, try to fix the issue by using the information provided by the error message. If this isn't possible or doesn't work, the core maintainers are here to help via a [GitHub Issue](https://github.com/EAPD-DRB/OG-ZAF/issues).

4. Now you're ready to [commit](https://help.github.com/articles/github-glossary/#commit) your changes to your local repo using the code below. The first line of code tells `Git` to track a file. Use "git status" to find all the files you've edited, and "git add" each of the files that you'd like `Git` to track. As a rule, do not add large files. If you'd like to add a file that is larger than 25 MB, please contact the other contributors and ask how to proceed. The second line of code commits your changes to your local repo and allows you to create a commit message; this should be a short description of your changes.

   *Tip*: Committing often is a good idea as `Git` keeps a record of your changes. This means that you can always revert to a previous version of your work if you need to. Do this to commit:
    ```
     OG-ZAF$ git add [filename]
     OG-ZAF$ git commit -m "[description-of-your-commit]"
    ```

5. Periodically, make sure that the branch you created in step 2 is in sync with the changes other contributors are making to the central master branch by fetching upstream and merging upstream/master into your branch:
    ```
      OG-ZAF$ git fetch upstream
      OG-ZAF$ git merge upstream/master
    ```
   You may need to resolve conflicts that arise when another contributor changed the same section of code that you are changing. Feel free to ask other contributors for guidance if this happens to you. If you do need to fix a merge conflict, re-run the test suite afterwards (step 4.)

6. When you are ready for other team members to review your code, make your final commit and push your local branch to your remote repo:
    ```
     OG-ZAF$ git push origin [new-branch-name]
    ```
7. From the GitHub.com user interface, [open a pull request](https://help.github.com/articles/creating-a-pull-request/#creating-the-pull-request).

8. When you open a GitHub pull request, a code coverage report will be automatically generated. If your branch adds new code that is not tested, the code coverage percent will decline and the number of untested statements ("misses" in the report) will increase. If this happens, you need to add to your branch one or more tests of your newly added code. Add tests so that the number of untested statements is the same as it is on the master branch.


(Sec_SimpleUsage)=
## Simple Usage

[TODO: Include simple usage examples.]


(Sec_ContribFootnotes)=
## Footnotes

[^recent_python]:The most recent version of Python from Anaconda is Python 3.12. `OG-ZAF` is currently tested to run on Python 3.10 and 3.11.

[^commandline_note]:The dollar sign is the end of the command prompt on a Mac. If you are using the Windows operating system, this is usually the right angle bracket (>). No matter the symbol, you don't need to type it (or anything to its left, which shows the current working directory) at the command line before you enter a command; the prompt symbol and preceding characters should already be there.
