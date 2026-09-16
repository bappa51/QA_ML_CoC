Prerequisite:
# Install numpy
# install pandas
# install matplotlib

command in Bash
pip3 install numpy pandas matplotlib

command from terminal
pip3 install numpy pandas matplotlib

if you are using Jupyter notebook
!pip install numpy pandas matplotlib

=================================================

CoQ Predictive Analytics Toolkit
This project provides a practical and transparent framework for analyzing Cost of Quality (CoQ) using release-level operational data. It combines multivariate linear regression with a Random Forest baseline to estimate CoQ, identify key drivers, and visualize relationships across releases.

The toolkit is designed for business and engineering teams who want to understand which factors most strongly influence CoQ and how predictive modeling can support operational decision-making. It is especially useful when working with structured data where rows represent operational axes and columns represent release observations.

What this project does
Reads CoQ data from CSV or Excel files
Automatically detects the sample dataset in the working folder
Builds a multivariate linear regression model
Builds a Random Forest comparison model
Evaluates model quality using leave-one-out cross-validation
Identifies the most influential CoQ drivers
Produces charts for:
actual vs predicted performance
feature importance
model comparison
residual analysis
correlation heatmaps
Why this matters
Cost of Quality is a key operational metric that reflects the balance between prevention, appraisal, and failure costs. By identifying the drivers of CoQ and measuring how strongly they relate to the target, organizations can prioritize interventions, reduce waste, and improve quality outcomes with greater confidence.

Typical use case
This toolkit is useful for teams analyzing:

process quality trends across releases
major contributors to CoQ
model-based forecasting of future quality costs
driver ranking for quality improvement programs
Getting started
Clone the repository
Create a virtual environment
Install dependencies
Run the analysis script
Review generated charts and output files
For detailed setup and usage instructions, see the project README and the run script included in this repository.

Optional polished tagline
Cost of Quality prediction and driver analysis using machine learning and release-level data.


