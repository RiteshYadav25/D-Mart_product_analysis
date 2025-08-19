# D-Mart_product_analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import pandas as pd
df = pd.read_csv("C:/Users/Ritesh Yadav/OneDrive/Desktop/DMart.csv")
df.shape
df.info()
df.head()
df[['SubCategory','BreadCrumbs']].tail(100)
df['Quantity'].unique()

df.duplicated().sum()
df.isnull().sum()

df['Name'] = df['Name'].fillna('Kitchen Appliance')
df['Brand'] = df['Brand'].fillna('Local/Unknown')
df[df['Price'].isnull()==True]

df['Price']=df['Price'].fillna(df[df['SubCategory']=='Personal Care/Nail Care']['Price'].mean())

df['DiscountedPrice']=df['DiscountedPrice'].fillna(df[df['SubCategory']=='Personal Care/Nail Care']['DiscountedPrice'].mean())
df[df['Category'].isnull()==True]

df['Category'] = df['Category'].fillna('Home & Kitchen')
df['SubCategory'] = df['SubCategory'].fillna('Home Appliances')
df[df.Quantity.isnull()==True]

df['Quantity'] = df['Quantity'].fillna('Size XXL')
df['Description'] = df['Description'].fillna('No Desciption')
df.isnull().sum()

df.head()
df['DiscountPercent'] = (df['Price']-df['DiscountedPrice'])*100/df['Price']
df.isnull().sum()

df[df['DiscountPercent'].isnull()==True]
df['DiscountPercent'] = df['DiscountPercent'].fillna(0)
df['DiscountPercent']=df['DiscountPercent'].apply(lambda x:int(x))
df.head()
df['SubCategory'].unique()
df['SubCategory'] = df['SubCategory'].apply(lambda x: x.split('/')[-1])
df['SubCategory'].unique()
df.head()
df.describe()
df[df['DiscountPercent']==98]

cols = ['Price','DiscountedPrice','DiscountPercent']

for i in cols:
    print('Distribution of',i)
    plt.figure(figsize=(20,6))
    sns.boxplot(x = df[i])
    plt.show()

for i in cols:
    print('Distribution of',i)
    plt.figure(figsize=(20,6))
    sns.histplot(x = df[i])
    plt.show()

df.head(2)

from wordcloud import WordCloud, STOPWORDS
text = "".join(d for d in df['Description'])
wc = WordCloud(width= 1600,height=800,colormap='prism',background_color = 'white').generate(text)
plt.figure(figsize = (30, 6))
plt.imshow(wc, interpolation="gaussian")
plt.axis("off")
plt.show()

df.Category.unique()
df.Category.value_counts()
df[(df['Category'] == 'Wonderchef') | (df['Category'] == 'Syska') | (df['Category'] == 'Butterfly') | (df['Category'] == 'Pigeon')| (df['Category'] == 'Zebronics') | (df['Category'] == 'Geep') | (df['Category'] == 'Joyo Plastics') ]
df['Category'] = df['Category'].replace({'Zebronics':'Appliances','Geep':'Appliances','Syska':'Home & Kitchen','Butterfly':'Home & Kitchen','Wonderchef':'Home & Kitchen','Joyo Plastics':'Home & Kitchen','Pigeon':'Home & Kitchen'})
df.Category.value_counts()

sns.set_context("poster",font_scale=0.5)
plt.figure(figsize=(20,10))
ax = df.Category.value_counts().plot(kind='bar',cmap='RdYlGn_r')

for p in ax.patches:
    ax.annotate(int(p.get_height()),(p.get_x()+0.25,p.get_height()+20),ha='center',color='black')
    
plt.title('Number of Products per Category',fontsize=18)
plt.xlabel('Categories')
plt.ylabel('Number of Products')

df['SubCategory']=df['SubCategory'].replace({'Zebronics':'Appliances','Geep':'Appliances','Syska':'Home Appliances','Butterfly':'Kitchen Appliances','Wonderchef':'Kitchen Appliances','Joyo Plastics':'Kitchen Appliances','Pigeon':'Kitchen Appliances'})
df.SubCategory.value_counts()
px.bar(x=df.SubCategory.value_counts().index, y=df.SubCategory.value_counts().values,color=df.SubCategory.value_counts().values,title='Number of Products in each Subcategory')
df.groupby("Category")[['SubCategory','Brand','Name']].nunique().rename(columns={"Name":'Number of Products'})

df.DiscountPercent.mean()
df.groupby('Category')['DiscountPercent'].mean().sort_values(ascending=False)
df[df['Category']=='DMart Grocery']
gro_Rice = df[(df['Category']=='Grocery')&(df['SubCategory']=='Rice & Rice Products')]
gro_Rice
DMart_gro_Rice = df[(df['Category']=='DMart Grocery')&(df['SubCategory']=='Rice & Rice Products')]
DMart_gro_Rice
print('DMart Rice avg Price', int(DMart_gro_Rice.Price.mean()))
print('Other Rice avg Price', int(DMart_gro_Rice.Price.mean()))

print('DMart Rice avg Discount', int(DMart_gro_Rice.DiscountPercent.mean()))
print('Other Rice avg Discount', int(DMart_gro_Rice.DiscountPercent.mean()))
text_gro_rice = "".join(t for t in gro_Rice['Description'])

gr_wc = WordCloud(width = 1000,height=500,margin=5,colormap='gist_rainbow',background_color='white').generate(text_gro_rice)

plt.imshow(gr_wc,interpolation='gaussian')
plt.axis('off')
plt.show()

text_dm_gro_rice = "".join(t for t in DMart_gro_Rice['Description'])

dgr_wc = WordCloud(width = 1000,height=500,margin=5,colormap='gist_rainbow',background_color='white').generate(text_dm_gro_rice)

plt.imshow(dgr_wc,interpolation='gaussian')
plt.axis('off')
plt.show()

df.groupby(['Category','SubCategory','Brand'])['Brand'].count()
clothing = df[df['Category']=='Clothing & Accessories']
clothing.SubCategory.value_counts()

px.pie(values= clothing.SubCategory.value_counts().values,names =clothing.SubCategory.value_counts().index,hole=0.6)

clothing.groupby('SubCategory')['DiscountPercent'].mean()

sns.set_context("poster",font_scale=0.5)
plt.figure(figsize=(20,10))
ax = clothing.groupby('Brand')['DiscountedPrice'].mean().plot(kind='bar',color='crimson',rot=0)

for p in ax.patches:
    ax.annotate(int(p.get_height()),(p.get_x()+0.25,p.get_height()-15),ha='center',color='white')
    
plt.title('Avg.DiscountedPrice for Brands in Clothing',fontsize=18)
plt.xlabel('Clothing Brands')
plt.ylabel('Avg.DiscountedPrice')

cloth_brand_mean = clothing.groupby(['Brand','SubCategory'])['DiscountedPrice'].mean().reset_index()
cloth_brand_mean

clothing[(clothing['SubCategory']=="Women's")&(clothing['DiscountPercent']>50)]
clothing[(clothing['SubCategory']=="Women's")&(clothing['DiscountPercent']>50)]
    df[df['Category']=='Personal Care'].groupby('SubCategory')[['Brand','Name']].nunique().sort_values(by='Name',ascending=False).rename(columns={'Name':'No.of.Products'})
    
