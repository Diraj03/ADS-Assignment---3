import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from sklearn import cluster
import sklearn.metrics as skmet

# Read the CSV file into a DataFrame


def read_file(file_path):
    """
    function takes name of the file and reads it from local 
    directory and loads this file into a dataframe. After that transposes the 
    dataframe and returns the original and transposed dataframes.

    Parameters
    ----------
    file_path : str
        a string containing path to the CSV datafile to be read.

    Returns
    -------
    data : pandas.DataFrame
        the resulting DataFrame read from the CSV file..

    """
    data = pd.read_csv(file_path)
    return data


file_path = r"C:\Users\Diraj\Downloads\API.csv"
data = read_file(file_path)

print(data)

# Filter data for specific indicators
data1 = data[data['Indicator Name'] == 'Population growth (annual %)']
data2 = data[data['Indicator Name'] ==
             'CO2 emissions from liquid fuel consumption (kt)']
data3 = data[data['Indicator Name'] ==
             'Electric power consumption (kWh per capita)']
data4 = data[data['Indicator Name'] ==
             'Methane emissions (kt of CO2 equivalent)']

# Drop unnecessary columns
data1 = data1.drop(["Indicator Code", "Indicator Name",
                   "Country Code", "2021", "2020"], axis=1)
# Replace NaN values with 0
data1 = data1.replace(np.NaN, 0)
# Filter specific countries
u = ["Benin", "Bangladesh", "Bahrain", "Brazil", "Colombia", "Canada"]
dt = data1["Country Name"].isin(u)
data1 = data1[dt]
# Print the filtered data
print(data1)

# Transpose the DataFrame
d_t = np.transpose(data1)
# Reset the index and rename columns
d_t = d_t.reset_index()
d_t = d_t.rename(columns=d_t.iloc[0])
d_t = d_t.drop(0, axis=0)
d_t = d_t.rename(columns={"Country Name": "Year"})

# Convert columns to numeric type
d_t["Year"] = pd.to_numeric(d_t["Year"])
d_t["Bahrain"] = pd.to_numeric(d_t["Bahrain"])
d_t["Brazil"] = pd.to_numeric(d_t["Brazil"])
d_t["Canada"] = pd.to_numeric(d_t["Canada"])
d_t["Colombia"] = pd.to_numeric(d_t["Colombia"])
d_t["Bangladesh"] = pd.to_numeric(d_t["Bangladesh"])
# Drop rows with NaN values
d_t = d_t.dropna()
# Print the modified DataFrame
print(d_t)
# Drop unnecessary columns
data2 = data2.drop(["Indicator Code", "Indicator Name",
                   "Country Code", "2021", "2020"], axis=1)
data2 = data2.replace(np.NaN, 0)
dt2 = data2["Country Name"].isin(u)
data2 = data2[dt2]
data2
# Transpose the DataFrame
trans2 = np.transpose(data2)
# Reset the index and rename columns
trans2 = trans2.reset_index()
trans2 = trans2.rename(columns=trans2.iloc[0])
trans2 = trans2.drop(0, axis=0)
trans2 = trans2.rename(columns={"Country Name": "Year"})
# Convert columns to numeric type
trans2["Year"] = pd.to_numeric(trans2["Year"])
trans2["Bahrain"] = pd.to_numeric(trans2["Bahrain"])
trans2["Brazil"] = pd.to_numeric(trans2["Brazil"])
trans2["Canada"] = pd.to_numeric(trans2["Canada"])
trans2["Colombia"] = pd.to_numeric(trans2["Colombia"])
trans2["Bangladesh"] = pd.to_numeric(trans2["Bangladesh"])
# Drop rows with NaN values
trans2 = trans2.dropna()
# Print the modified DataFrame
print(trans2)

# Drop unnecessary columns
data3 = data3.drop(["Indicator Code", "Indicator Name",
                   "Country Code", '2018', '2019', "2021", "2020"], axis=1)
data3 = data3.replace(np.NaN, 0)
dt3 = data3["Country Name"].isin(u)
data3 = data3[dt3]
# Transpose the DataFrame
trans3 = np.transpose(data3)
# Reset the index and rename columns
trans3 = trans3.reset_index()
trans3 = trans3.rename(columns=trans3.iloc[0])
trans3 = trans3.drop(0, axis=0)
trans3 = trans3.rename(columns={"Country Name": "Year"})
# Convert columns to numeric type
trans3["Year"] = pd.to_numeric(trans3["Year"])
trans3["Bahrain"] = pd.to_numeric(trans3["Bahrain"])
trans3["Brazil"] = pd.to_numeric(trans3["Brazil"])
trans3["Canada"] = pd.to_numeric(trans3["Canada"])
trans3["Colombia"] = pd.to_numeric(trans3["Colombia"])
trans3["Bangladesh"] = pd.to_numeric(trans3["Bangladesh"])
# Drop rows with NaN values
trans3 = trans3.dropna()
# Print the modified DataFrame
print(trans3)

# Drop unnecessary columns
data4 = data4.drop(["Indicator Code", "Indicator Name",
                   "Country Code", "2021", "2020"], axis=1)
data4 = data4.replace(np.NaN, 0)
dt4 = data4["Country Name"].isin(u)
data4 = data4[dt4]
# Transpose the DataFrame
trans4 = np.transpose(data4)
# Reset the index and rename columns
trans4 = trans4.reset_index()
trans4 = trans4.rename(columns=trans4.iloc[0])
trans4 = trans4.drop(0, axis=0)
trans4 = trans4.rename(columns={"Country Name": "Year"})
# Convert columns to numeric type
trans4["Year"] = pd.to_numeric(trans4["Year"])
trans4["Bahrain"] = pd.to_numeric(trans4["Bahrain"])
trans4["Brazil"] = pd.to_numeric(trans4["Brazil"])
trans4["Canada"] = pd.to_numeric(trans4["Canada"])
trans4["Colombia"] = pd.to_numeric(trans4["Colombia"])
trans4["Bangladesh"] = pd.to_numeric(trans4["Bangladesh"])
# Drop rows with NaN values
trans4 = trans4.dropna()
# Print the modified DataFrame
print(trans4)

# Create a DataFrame for Brazil
Brazil = pd.DataFrame()
Brazil["Year"] = d_t["Year"]
Brazil["Population growth"] = d_t["Brazil"]
Brazil["co2_emission"] = trans2["Brazil"]
Brazil["Electric_power_consumption"] = trans3["Brazil"]
Brazil["Methane emissions"] = trans4["Brazil"]
Brazil = Brazil.iloc[1:57, :]

# Create a DataFrame for Canada
Canada = pd.DataFrame()
Canada["Year"] = d_t["Year"]
Canada["Population growth"] = d_t["Canada"]
Canada["co2_emission"] = trans2["Canada"]
Canada["Electric_power_consumption"] = trans3["Canada"]
Canada["Methane emissions"] = trans4["Canada"]
Canada = Canada.iloc[1:57, :]


# Function to plot scatter matrix for a given country DataFrame
def set_mat(country):
    """
    This function takes in a pandas dataframe containing information about a 
    country and plots a scatter matrix of the variables in the dataframe. 
    It then displays the plot to the user.

    Parameters
    ----------
    country : pandas.DataFrame
        dataframe containing information about a country for the respective 
        index.

    Returns
    -------
    plot Scatter matrix for the indicator

    """
    pd.plotting.scatter_matrix(country, figsize=(14.0, 12.0))
    plt.tight_layout()
    plt.show()


# Calling the function for Canada
set_mat(Canada)
# Calling the function for Brazil
set_mat(Brazil)

# extract columns for fitting.
# .copy() prevents changes in the data
df = Brazil[["Electric_power_consumption", "co2_emission"]].copy()

# normalise dataframe and inspect result
# normalisation is done only on the extract columns. .copy() prevents
print(df.describe())

print("n   score")
# loop over trial numbers of clusters calculating the silhouette
for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df)

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print(ic, skmet.silhouette_score(df, labels))


cepc = np.array(Canada["Electric_power_consumption"]).reshape(-1, 1)
cco2 = np.array(Canada["co2_emission"]).reshape(-1, 1)

cl = np.concatenate((cepc, cco2), axis=1)
nc = 4
# fitting the model
kmeans = cluster.KMeans(n_clusters=nc)
kmeans.fit(cl)
# assignining the label
label = kmeans.labels_
# finding the centers for cluster
km_c = kmeans.cluster_centers_
col = ["Electric_power_consumption", 'co2_emission']
labels = pd.DataFrame(label, columns=['Cluster ID'])
result = pd.DataFrame(cl, columns=col)
# concat result and labels
res = pd.concat((result, labels), axis=1)
# plotting the cluster
plt.figure(figsize=(7.0, 7.0))
plt.title("Canada Electric_power_consumption vs co2_emission ", fontweight='bold')
plt.scatter(res["Electric_power_consumption"],
            res["co2_emission"], c=label, cmap="tab10")
plt.scatter(km_c[:, 0], km_c[:, 1], marker="*", c="black", s=200)
plt.xlabel("Electric_power_consumption", fontweight='bold', fontsize=12)
plt.ylabel("co2_emission", fontweight='bold', fontsize=12)
# plotting centers of clusters
plt.show()

# Reshaping the columns
epc = np.array(Brazil["Electric_power_consumption"]).reshape(-1, 1)
co2 = np.array(Brazil["co2_emission"]).reshape(-1, 1)
cl = np.concatenate((epc, co2), axis=1)
nc = 4
# fitting the model
kmeans = cluster.KMeans(n_clusters=nc)
kmeans.fit(cl)
# assignining the label
label = kmeans.labels_
# finding the centers for cluster
km_c = kmeans.cluster_centers_
col = ["Electric_power_consumption", 'co2_emission']
labels = pd.DataFrame(label, columns=['Cluster ID'])
result = pd.DataFrame(cl, columns=col)
# concat result and labels
res = pd.concat((result, labels), axis=1)
# plotting the cluster
plt.figure(figsize=(7.0, 7.0))
plt.title("Brazil Electric_power_consumption vs co2_emission ", fontweight='bold')
plt.scatter(res["Electric_power_consumption"],
            res["co2_emission"], c=label, cmap="tab10")
plt.scatter(km_c[:, 0], km_c[:, 1], marker="*", c="black", s=200)
plt.xlabel("Electric_power_consumption", fontweight='bold', fontsize=12)
plt.ylabel("co2_emission", fontweight='bold', fontsize=12)
# plotting centers of clusters
plt.show()

# Normalizing the value of co2_emission
Canada["norm_CO2_emission"] = Canada["co2_emission"] / \
    Canada["co2_emission"].abs().max()
print(Canada)

# Normalizing the value of co2_emission
Brazil["norm_CO2_emission"] = Brazil["co2_emission"] / \
    Brazil["co2_emission"].abs().max()
print(Brazil)

# FFunction to calculate the error range


def err_ranges(x, f, param, sigma):
    """
    This function calculates the lower and upper limits for a given set of 
    parameters and their corresponding errors.

    Parameters
    ----------
    x : 1D numpy array
        input array to be passed to the function
    f : function object
        function to be fitted to the data.
    param : tuple
        array of parameters obtained from fitting.
    sigma : tuple
        array of errors corresponding to the parameters.

    Returns
    -------
    which gives the lower and upper bounds of the function at each x value

    """

    import itertools as iter

    # initiate arrays for lower and upper limits
    lower = f(x, *param)
    upper = lower

    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        nmin = p - s
        nmax = p + s
        uplow.append((nmin, nmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = f(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)

    return lower, upper

# Function to calculate the low order polynominals


def poly(t, p0, p1, p2, p3):
  """
  function takes in an input array u representing years from 1990 and returns
  polynomial function.

  Parameters
  ----------
  t : 
      input array.
  p0,p1,p2,p3 : 
      representing the coefficients of a polynomial function.

  Returns
  -------
  returns the corresponding values of the polynomial function.

  """
  t = t - 1990
  f = p0 + p1*t + p2*t**2 + p3*t**3
  return f


# Use the curve_fit function to fit a curve to the data
pop, pcorr = opt.curve_fit(poly, Brazil["Year"], Brazil["norm_CO2_emission"])
print("Fit parameter", pop)
# extract variances and calculate sigmas
sig = np.sqrt(np.diag(pcorr))
# call function to calculate upper and lower limits with extrapolation
# create extended year range
years = np.arange(1980, 2036)
lower, upper = err_ranges(years, poly, pop, sig)
Brazil["poly"] = poly(Brazil["Year"], *pop)
plt.figure(figsize=(15, 8))
plt.title("Brazil CO2_emission prediction till year 2036")
plt.plot(Brazil["Year"], Brazil["norm_CO2_emission"], label="data")
plt.plot(Brazil["Year"], Brazil["poly"], label="fit")
# Labels for x and y axis
plt.xlabel("Years", fontweight='bold', fontsize=12)
plt.ylabel("co2_emission", fontweight='bold', fontsize=12)
# plot error ranges with transparency
plt.fill_between(years, lower, upper, alpha=0.5, label='Prediction till 2036')
plt.legend(loc="upper left", fontsize=15)
plt.show()

# Use the curve_fit function to fit a curve to the data
popt, pcorr = opt.curve_fit(poly, Canada["Year"], Canada["norm_CO2_emission"])
print("Fit parameter", popt)
# extract variances and calculate sigmas
sig = np.sqrt(np.diag(pcorr))
# call function to calculate upper and lower limits with extrapolation
# create extended year range
years = np.arange(1980, 2036)
lower, upper = err_ranges(years, poly, popt, sig)
Canada["poly"] = poly(Canada["Year"], *popt)
plt.figure(figsize=(15, 8))
plt.title("Canada CO2_emission prediction till year 2036")
plt.plot(Canada["Year"], Canada["norm_CO2_emission"], label="data")
plt.plot(Canada["Year"], Canada["poly"], label="fit")
# Labels for x and y axis
plt.xlabel("Years", fontweight='bold', fontsize=12)
plt.ylabel("co2_emission", fontweight='bold', fontsize=12)
# plot error ranges with transparency
plt.fill_between(years, lower, upper, alpha=0.5, label='Prediction till 2036')
plt.legend(loc="upper left", fontsize=15)
plt.show()

# Function to calculate the best fitting and the conifidence range


def cubic(x, a, b, c, d):
    """Returns cubic function a*x^3 + b*x^2 + c*x + d"""

    f = a*x**3 + b*x**2 + c*x + d

    return f


# To plot the best fitting function and the conifidence range for canada
param, covar = opt.curve_fit(
    cubic, Canada["Year"], Canada["norm_CO2_emission"])
# create monotonic x-array for plotting
x = Canada["Year"]
y = cubic(Canada["Year"], *param)
y_pred = poly(Canada["Year"], *popt)

# Calculate the confidence range
y_err = Canada["norm_CO2_emission"] - y_pred
mse = np.mean(y_err ** 2)
n = len(x)
conf_interval = 1.96 * np.sqrt(mse / n)  # 1.96 for 95% confidence interval
plt.figure(figsize=(15, 8))

plt.plot(Canada["Year"], Canada["norm_CO2_emission"],
         "o", markersize=3, label="Data")
plt.plot(x, y, color='red', label='Fitted Function')
# Labels for x and y axis
plt.xlabel("Years", fontweight='bold', fontsize=12)
plt.ylabel("co2_emission", fontweight='bold', fontsize=12)
plt.fill_between(Canada["Year"], y_pred - conf_interval, y_pred +
                 conf_interval, color='gray', alpha=0.5, label='Confidence Range')
plt.title("Best fitting function and the conifidence range for canada", fontsize=12)
plt.legend(fontsize=12)
plt.show()

# To plot the best fitting function and the conifidence range for Brazil
param, covar = opt.curve_fit(
    cubic, Brazil["Year"], Brazil["norm_CO2_emission"])
# create monotonic x-array for plotting
x = Brazil["Year"]
y = cubic(Brazil["Year"], *param)
y_pred = poly(Brazil["Year"], *pop)

# Calculate the confidence range
y_err = Brazil["norm_CO2_emission"] - y_pred
mse = np.mean(y_err ** 2)
n = len(x)
conf_interval = 1.96 * np.sqrt(mse / n)  # 1.96 for 95% confidence interval
# plotting the line graph
plt.figure(figsize=(15, 8))
plt.plot(Brazil["Year"], Brazil["norm_CO2_emission"],
         "o", markersize=3, label="Data")
plt.plot(x, y, color='red', label='Fitted Function')
# Labels for x and y axis
plt.xlabel("Years", fontweight='bold', fontsize=12)
plt.ylabel("co2_emission", fontweight='bold', fontsize=12)
plt.fill_between(Brazil["Year"], y_pred - conf_interval, y_pred +
                 conf_interval, color='gray', alpha=0.5, label='Confidence Range')
plt.title("Best fitting function and the conifidence range for brazil", fontsize=12)
plt.legend(fontsize=12)
plt.show()
