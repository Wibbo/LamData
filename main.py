import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('A diamond price use case')
st.markdown('Using Python to provide predicted outcomes for target attributes (dependent variables). In this case, we want to be able to predict diamond prices give a set of associated attributes.')

st.markdown('## Dataset attributes')
st.markdown('* price: The cost of each diamond in the dataset')
st.markdown('* carat: The weight of each diamond in the dataset')
st.markdown('* cut: The quality of the cut (fair, good, very good, premium or ideal)')
st.markdown('* colour: The perceived colour quality from J (worst) to D (best)')
st.markdown('* clarity: The clearness of the diamond (categorical values)')
st.markdown('* x: The diamond length in mm')
st.markdown('* y: The diamond width in mm')
st.markdown('* z: The diamond depth in mm')
st.markdown('* depth: The depth when comparing to the length and width 2*z/(x + y)')
st.markdown('* table: The width of the top of the diamond compared to its widest point')

dmd = pd.read_csv('./diamond.csv')

# Determine some basic information about the data.
price_IQR = dmd.price.quantile(0.75) - dmd.price.quantile(0.25)
weight_IQR = dmd.carat.quantile(0.75) - dmd.carat.quantile(0.25)
st.markdown('---')
st.markdown('## Basic details for the dataset')
st.markdown(f'* There are {dmd.shape[0]} observations, with {dmd.shape[1]} attributes, in the diamond dataset.')
st.markdown(f'* The IQR for prices in this dataset is {price_IQR:.1f}')
st.markdown(f'* The price IQR * 1.5 is {price_IQR*1.5:.1f}')

st.markdown(f'* The IQR for carats in this dataset is {weight_IQR}')
st.markdown(f'* The carat IQR * 1.5 is {weight_IQR*1.5:.2f}')
st.markdown('---')
st.markdown('## A few sample records from the diamond dataset')
st.write(dmd.head())
st.markdown('---')
st.markdown('## Basic statistics for the dataset')
st.write(dmd.describe())
st.markdown('Note the high values of the y and z dimensions (I would expect a 6cm diamond to cost more than £18k)')
st.markdown('A price histogram may also provide further insight.')

fig1, axes = plt.subplots(1, 1, figsize=(15, 9))
ax = sns.histplot(data=dmd, kde=True, binwidth=2000, x='price')
st.pyplot(fig1)

# The max value for the carat attribute seems high...
st.markdown('---')
st.markdown('## Outliers')
st.markdown('### Boxplots for diamond prices and weights')
fig2, axes = plt.subplots(2, 1, figsize=(15,12))
ax = sns.boxplot(ax=axes[0], x=dmd['price']).set_title('Diamond prices')
ax = sns.boxplot(ax=axes[1], x=dmd['carat']).set_title('Diamond weights')
st.markdown('Consider how to treat outliers and whether to remove them from the dataset.')
st.pyplot(fig2)

st.markdown('## Diamond dimensions')
st.markdown('The x, y, z values represent the physical dimensions of each diamond.')
st.markdown('Since zero is not a valid measurement, we assume that zero entries represent errors.')

zero_x = len(dmd.loc[dmd['x']==0])
st.markdown(f'Observations where x is zero: {zero_x}')

zero_y = len(dmd.loc[dmd['y']==0])
st.markdown(f'Observations where y is zero: {zero_y}')

zero_z = len(dmd.loc[dmd['z']==0])
st.markdown(f'Observations where z is zero: {zero_z}')

st.markdown('For now, we will remove x, y, z zero values from the dataset.')
dmd = dmd.loc[dmd['x']>0]
dmd = dmd.loc[dmd['y']>0]
dmd = dmd.loc[dmd['z']>0]

zero_x = len(dmd.loc[dmd['x']==0])
st.markdown(f'Observations where x is zero: {zero_x}')

zero_y = len(dmd.loc[dmd['y']==0])
st.markdown(f'Observations where y is zero: {zero_y}')

zero_z = len(dmd.loc[dmd['z']==0])
st.markdown(f'Observations where z is zero: {zero_z}')
st.markdown(f'There are now {dmd.shape[0]} remaining observations.')

st.markdown('## Heat map for the diamond dataset')
fig3, axes = plt.subplots(1, 1, figsize=(15,9))
cor = dmd.corr()
sns.heatmap(cor, annot=True, fmt='.2f')
st.pyplot(fig3)