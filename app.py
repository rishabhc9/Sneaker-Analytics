import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import datetime as dt
import pickle
from contextlib import suppress

st.set_page_config(
    page_title="Sneaker Analytics",
    page_icon="👾",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main_page():
    global df
    st.title('Sneaker Sales')
    original_data = pd.read_csv('/Users/rishabhchopda/Downloads/DA Project/StockX-Data-Contest-2019-3.csv')
    df = original_data.copy()

    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Release Date'] = pd.to_datetime(df['Release Date'])
    df['Sneaker Name'] = df['Sneaker Name'].apply(lambda x: x.replace('-', ' '))
    obj_cols = ['Sale Price','Retail Price']
    for col in obj_cols:
        df[str(col)] = pd.to_numeric(df[str(col)])

    df['Bought for Less Than Retail'] = df['Sale Price'] < df['Retail Price']
    df['Bought for Retail'] = df['Sale Price'] == df['Retail Price']
    df['Bought for More Than Retail'] = df['Sale Price'] > df['Retail Price']

    
    df_cat = ['Release Date', 'Buyer Region', 'Sneaker Name', 'Retail Price', 'Shoe Size', 'Brand', 'Bought for Retail', 'Bought for Less Than Retail', 'Bought for More Than Retail' ]
    for cat in df_cat:
        cat_num = df[str(cat)].value_counts()
        plt.figure(figsize=(15,6))
        chart = sns.barplot(x = cat_num.index, y= cat_num)
        chart.set_title("Sneakers Sales by %s" % (cat))
        plt.ylabel("Sneaker Sales")
        chart.set_xticklabels(chart.get_xticklabels(), rotation = 90)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
        fig=chart.figure
        fig.patch.set_facecolor('#0f1116')
        plt.rcParams.update({'axes.facecolor':'#0f1116'})
        plt.figure(facecolor='#0f1116') 
        COLOR = 'white'
        plt.rcParams['text.color'] = COLOR
        plt.rcParams['axes.labelcolor'] = COLOR
        plt.rcParams['xtick.color'] = COLOR
        plt.rcParams['ytick.color'] = COLOR

        plt.rc('axes',edgecolor=COLOR)
        st.pyplot(fig)

def page2():
    original_data = pd.read_csv('/Users/rishabhchopda/Downloads/DA Project/StockX-Data-Contest-2019-3.csv')
    df = original_data.copy()
    st .title('Sneaker Name vs Sale Price')
    temp = df[['Sneaker Name', 'Sale Price']]

    # Clean up this list
    sneakernames = ['adidas Yeezy Boost 350 V2 Butter',
        'Adidas Yeezy Boost 350 V2 Beluga 2pt0',
        'Adidas Yeezy Boost 350 V2 Zebra',
        'Adidas Yeezy Boost 350 V2 Blue Tint',
        'Adidas Yeezy Boost 350 V2 Cream White',
        'Adidas Yeezy Boost 350 V2 Sesame', 'adidas Yeezy Boost 350 V2 Static',
        'Adidas Yeezy Boost 350 V2 Semi Frozen Yellow',
        'Air Jordan 1 Retro High Off White University Blue',
        'Adidas Yeezy Boost 350 V2 Static Reflective',
        'Nike Air Presto Off White Black 2018',
        'Nike Air Presto Off White White 2018',
        'Nike Air VaporMax Off White 2018',
        'Nike Blazer Mid Off White All Hallows Eve',
        'Nike Blazer Mid Off White Grim Reaper', 'Nike Zoom Fly Off White Pink',
        'Nike Air VaporMax Off White Black',
        'Nike Zoom Fly Off White Black Silver',
        'Nike Air Force 1 Low Off White Volt',
        'Adidas Yeezy Boost 350 V2 Core Black Red 2017',
        'Nike Air Force 1 Low Off White Black White',
        'Air Jordan 1 Retro High Off White Chicago',
        'Nike Air Max 90 Off White Black',
        'Nike Zoom Fly Mercurial Off White Total Orange',
        'Nike Air Max 90 Off White Desert Ore',
        'Nike Zoom Fly Mercurial Off White Black', 'Nike Air Max 90 Off White',
        'Adidas Yeezy Boost 350 V2 Core Black White',
        'Nike Air Presto Off White', 'Nike Air Max 97 Off White',
        'Nike Air VaporMax Off White', 'Nike Blazer Mid Off White',
        'Adidas Yeezy Boost 350 Low V2 Beluga',
        'Nike React Hyperdunk 2017 Flyknit Off White',
        'Nike Air Force 1 Low Off White', 'Nike Zoom Fly Off White',
        'Nike Air Max 97 Off White Menta',
        'Air Jordan 1 Retro High Off White White',
        'Adidas Yeezy Boost 350 V2 Core Black Red',
        'Nike Air Max 97 Off White Black',
        'Nike Blazer Mid Off White Wolf Grey',
        'Adidas Yeezy Boost 350 V2 Core Black Copper',
        'Nike Air Max 97 Off White Elemental Rose Queen',
        'Adidas Yeezy Boost 350 V2 Core Black Green',
        'Adidas Yeezy Boost 350 Low Pirate Black 2016',
        'Adidas Yeezy Boost 350 Low Moonrock',
        'Adidas Yeezy Boost 350 Low Pirate Black 2015',
        'Adidas Yeezy Boost 350 Low Oxford Tan',
        'Adidas Yeezy Boost 350 Low Turtledove',
        'Nike Air Force 1 Low Virgil Abloh Off White AF100'
        ]
    avgs = []
    for name in sneakernames:
        shoerow = temp.loc[temp['Sneaker Name'] == name]
        avgs.append(shoerow.mean()[0])
    AvgPrice = pd.Series(avgs)
    SneakerName = pd.Series(sneakernames)
    avgprice_df = pd.DataFrame(columns = ['Sneaker_Name', 'Average_Price'])
    avgprice_df['Sneaker_Name'] = SneakerName
    avgprice_df['Average_Price'] = AvgPrice

    # Crerating visual of average shoe price
    fig_dims = (15, 4)
    fig, ax = plt.subplots(figsize=fig_dims)
    chart = sns.barplot(x = avgprice_df['Sneaker_Name'] , y= avgprice_df['Average_Price'])
    chart.set_xticklabels(chart.get_xticklabels(), rotation = 90)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

    fig.patch.set_facecolor('#0f1116')
    plt.rcParams.update({'axes.facecolor':'#0f1116'})
    plt.figure(facecolor='#0f1116') 
    
    COLOR='white'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR

    plt.rc('axes',edgecolor=COLOR)
    fig=chart.figure
    st.pyplot(fig)

#--------------------------------------------------------------------------------------------------------------------------------------------------------
    brg = df[['Buyer Region', 'Sale Price']]
    unq_brgs = df['Buyer Region'].value_counts().index.tolist()
    avg_5 = []

    for region in unq_brgs:
        regionrow = brg.loc[brg['Buyer Region'] == str(region)]
        avg_5.append(regionrow['Sale Price'].mean())

    unq_regions = pd.Series(unq_brgs)
    region_avgs = pd.Series(avg_5)
    regionprice_df = pd.DataFrame(columns = ['Buyer Region', 'Average Price'])
    regionprice_df['Buyer Region'] = unq_regions.sort_values(ascending = True)
    regionprice_df['Average Price'] = region_avgs

    fig_dims = (11, 10)
    fig1, ax = plt.subplots(figsize=fig_dims)
    chart2 = sns.barplot(x="Average Price", y="Buyer Region", data=regionprice_df)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(20))

    fig1.patch.set_facecolor('#0f1116')
    plt.rcParams.update({'axes.facecolor':'#0f1116'})
    plt.figure(facecolor='#0f1116') 
    
    COLOR='white'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR

    plt.rc('axes',edgecolor=COLOR)

    chart2.set_title("Average Sale Price by Buyer Region")
    fig2=chart2.figure
    st.pyplot(fig2)
#--------------------------------------------------------------------------------------------------------------------------------------------------------
    avgs_2 = []
    bds = df[['Brand', 'Sale Price']]
    brands = [' Yeezy', 'Off-White']
    for brand in brands:
        brandrow = bds.loc[bds['Brand'] == str(brand)]
        avgs_2.append(brandrow['Sale Price'].mean())
    col1, col2 = st.columns(2)
    col1.metric("Yeezy average price ($)", str(avgs_2[0]))
    col2.metric("Off-White average price ", str(avgs_2[1]))

def page3():
    with suppress(Exception):
        st.title('Sneaker Resale Price Prediction')
        model = pickle.load(open("/Users/rishabhchopda/Downloads/DA Project/model.pkl", "rb"))
        shoe_data = pd.read_csv("/Users/rishabhchopda/Downloads/DA Project/StockX-Data-Contest-2019-3.csv",parse_dates=True,)
        df = shoe_data.copy()
        df = df.drop(["Sale Price"], axis="columns")
        df = df.rename(
            columns={
                "Order Date": "Order_date",
                "Sneaker Name": "Sneaker_Name",
                "Retail Price": "Retail_Price",
                "Release Date": "Release_Date",
                "Shoe Size": "Shoe_Size",
                "Buyer Region": "Buyer_Region",
            }
        )

    # Taking Input -------------------------------------------------------------------------------------
        od = st.date_input("Order Date",dt.date(2030, 2, 13),key = 'odt')
        od=od.strftime("%m/%d/%y")

        brand = 'Off-White'

        sneakernames = ['Air Jordan 1 Retro High Off White University Blue',
        'Nike Air Presto Off White Black 2018',
        'Nike Air Presto Off White White 2018',
        'Nike Air VaporMax Off White 2018',
        'Nike Blazer Mid Off White All Hallows Eve',
        'Nike Blazer Mid Off White Grim Reaper', 'Nike Zoom Fly Off White Pink',
        'Nike Air VaporMax Off White Black',
        'Nike Zoom Fly Off White Black Silver',
        'Nike Air Force 1 Low Off White Volt',
        'Nike Air Force 1 Low Off White Black White',
        'Air Jordan 1 Retro High Off White Chicago',
        'Nike Air Max 90 Off White Black',
        'Nike Zoom Fly Mercurial Off White Total Orange',
        'Nike Air Max 90 Off White Desert Ore',
        'Nike Zoom Fly Mercurial Off White Black', 'Nike Air Max 90 Off White',
        'Nike Air Presto Off White', 'Nike Air Max 97 Off White',
        'Nike Air VaporMax Off White', 'Nike Blazer Mid Off White',
        'Nike React Hyperdunk 2017 Flyknit Off White',
        'Nike Air Force 1 Low Off White', 'Nike Zoom Fly Off White',
        'Nike Air Max 97 Off White Menta',
        'Air Jordan 1 Retro High Off White White',
        'Nike Air Max 97 Off White Black',
        'Nike Blazer Mid Off White Wolf Grey',
        'Nike Air Max 97 Off White Elemental Rose Queen',
        'Nike Air Force 1 Low Virgil Abloh Off White AF100']

        sneaker_name = st.selectbox('Sneaker Name',sneakernames)

        retail_price = st.number_input('Retail Price ($)',value=350)

        rd = st.date_input("Release",dt.date(2018, 8, 13),key = 'rdt')
        rd=rd.strftime("%m/%d/%y")

        shoesizes=[3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0,12.5,13.0,13.5,14.0,14.5,15.0,16.0,17.0]


        shoe_size = st.selectbox('Shoe Size: ',shoesizes)

        buyer_reg = st.text_input('Buyer Region (US States)', 'Alaska')
        
        

    # Taking Input -------------------------------------------------------------------------------------

        features = [od,brand,sneaker_name,retail_price,rd,shoe_size,buyer_reg]
        cols = [
            "Order_date",
            "Brand",
            "Sneaker_Name",
            "Retail_Price",
            "Release_Date",
            "Shoe_Size",
            "Buyer_Region",
        ]

        input_dictionary = dict(zip(cols, features))
        data = df.append(input_dictionary, ignore_index=True)

        # Converting dates into numericals

        data["Order_date"] = pd.to_datetime(data["Order_date"])
        data["Order_date"] = data["Order_date"].map(dt.datetime.toordinal)

        data["Release_Date"] = pd.to_datetime(data["Release_Date"], errors="coerce")
        data["Release_Date"] = data["Release_Date"].map(dt.datetime.toordinal)

        # Getting rid of null values
        data = data.dropna()

        object_cols = ["Sneaker_Name", "Buyer_Region", "Brand"]
        # Apply one-hot encoder to each column with categorical data
        OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(data[object_cols]))

        # One-hot encoding removed index; put it back
        OH_cols_train.index = data.index

        # Adding the column names after one hot encoding
        OH_cols_train.columns = OH_encoder.get_feature_names(object_cols)

        # Remove categorical columns (will replace with one-hot encoding)
        num_data = data.drop(object_cols, axis=1)

        # Add one-hot encoded columns to numerical features
        bigdata = pd.concat([num_data, OH_cols_train], axis=1)

        result=st.button('Predict')
        if result:
            prediction = model.predict(bigdata[-1:])
            pred_op= round(prediction[0], 2)
            st.success(f'Predicted Price: {pred_op}$')


page_names_to_funcs = {
"Sneaker Sales": main_page,
"Sneaker Name vs Sale Price": page2,
"Sneaker Resale Price Prediction": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
