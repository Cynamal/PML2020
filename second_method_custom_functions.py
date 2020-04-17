%%writefile custom_tools_2.py
import pandas as pd
import numpy as np
from collections import defaultdict 

dict_additional_columns = {
#     'color_RED': ['RED'],
#     'color_PINK': ['PINK', 'FUSHIA'],
#     'color_BLUE': ['BLUE'],
#     'color_GREEN': ['GREEN'],
#     'color_WHITE': ['WHITE', 'WHIT'],
#     'color_BLACK': ['BLACK', 'BLK', 'BLCK'],
#     'color_CREAM': ['CREAM'],
#     'color_GOLD': ['GOLD'],
#     'color_SILVER': ['SILVER'],
#     'color_COOPER': ['COOPER'],
#     'color_ASSORTED': ['ASSORTED'],
    'cat_POSTAGE': ['POSTAGE'],
    'cat_SAMPLES': ['SAMPLES'],
    'cat_MANUAL': ['Manual'],
    'cat_FEES': ['Bank Charges', 'bank charges', 'AMAZON FEE'],
#     'cat_BAD_DEBT': ['bad debt'],
#     'cat_SET': ['SET'],
    'material_WOOD': ['WOOD'],
    'material_CERAMIC': ['CERAMIC'],
    'material_GLASS': ['GLASS'],
#     'material_GEM': ['AMETHYST', 'DIAMANTE', 'RUBY', 'AMBER', 'TURQUISE', 'QUARTZ', 'GEMSTONE', 'CRYSTAL', 'JADE'],
#     'material_ENAMEL': ['ENAMEL'],
#     'material_METAL': ['COOPER', 'ZINC', 'BRONZE'],
    'style_RETRO': ['RETRO'],
    'style_VINTAGE': ['VINTAGE'],
    'style_HISTORIC': ['EDWARDIAN', 'FRENCH', 'BAROQUE', 'MOROCCAN', 'ANTIQUE'],
    'style_MODERN': ['MODERN', 'SCANDINAVIAN'],
    'type_JEWELRY': ['NECKLAGE', 'BEAD', 'RING', 'JEWEL', 'BRACELET'],
    'type_CHRISTMAS': ['CHRISTMAS'],
#     'type_BAG': ['BAG'],
#     'type_CONTAINER': ['TIN', 'BOX', 'CHEST', 'JAR']
}
dict_additional_columns.keys()

countries = ['Saudi Arabia', 'Czech Republic', 'Nigeria', 'Bermuda', 'West Indies', 'Lebanon', 'European Community',
           'Korea', 'Thailand', 'Brazil']

def get_feats(df):
    feats = df.select_dtypes([np.number, np.bool]).columns
    black_list = ['is_canceled', 'is_test', 'price_unit', 'cnt_p_product_orders']
    return [x for x in feats if x not in black_list]

def get_invoice_date_parameters(df):
    df['year'] = df['invoice_date'].dt.year
    df['month'] = df['invoice_date'].dt.month
    #df['day'] = df['invoice_date'].dt.day
    df['hour'] = df['invoice_date'].dt.hour
    #df['minute'] = df['invoice_date'].dt.minute
    df['day_of_year'] = df['invoice_date'].dt.dayofyear
    df['day_of_week'] = df['invoice_date'].dt.dayofweek
    df['week_of_year'] = df['invoice_date'].dt.weekofyear
    #df['quarter'] = df['invoice_date'].dt.quarter
    df['weekend'] = np.where(df['day_of_week'] < 5, 0, 1)
    return df

def one_hot_encoding(df, column):
    df[column] = pd.Categorical(df[column])
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
    return df

def get_additional_bool_column_from_description(df):
    df['description'] = df['description'].astype(str)
    for column, words in dict_additional_columns.items():
        df[column] = df['description'].apply(lambda x: any([word in x for word in words]))
    return df

def prepare_features(df_all):
    def group_to_dict(group_key, agg_func=np.sum):
        train = df_all[ ~df_all['is_canceled'].isnull()]
        dict_ = train.groupby(group_key)['is_canceled'].agg(agg_func).to_dict()
        if -1 in dict_: 
            del dict_[-1]
        mean = np.mean( list(dict_.values()) )
        return defaultdict(lambda: mean, dict_)    
    df_all['cnt_customer_cancel'] = df_all['customer_id'].map(group_to_dict('customer_id')).astype('float64')
    df_all['cnt_customer_orders'] = df_all['customer_id'].map(group_to_dict('customer_id', agg_func=np.size))
    df_all['ratio_customer_orders'] = (df_all['cnt_customer_cancel']/df_all['cnt_customer_orders']).round(5)
    
    #df_all['cnt_product_cancel'] = df_all['stock_code'].map(group_to_dict('customer_id')).astype('float64')
    #df_all['cnt_product_cancel_country'] = df_all['stock_code'].map(group_to_dict(['customer_id', 'country'])).astype('float64')
    #df_all['cnt_product_orders'] = df_all['stock_code'].map(group_to_dict('customer_id', agg_func=np.size))
    #df_all['ratio_product_orders'] = (df_all['cnt_product_cancel']/df_all['cnt_product_orders']).round(5)
    
    df_all['cnt_p_product_cancel'] = df_all['stock_code'].map(group_to_dict('stock_code')).astype('float64')
    #df_all['cnt_p_product_cancel_country'] = df_all['stock_code'].map(group_to_dict(['stock_code', 'country'])).astype('float64')
    df_all['cnt_p_product_orders'] = df_all['stock_code'].map(group_to_dict('stock_code', agg_func=np.size))
    df_all['ratio_p_product_orders'] = (df_all['cnt_p_product_cancel']/df_all['cnt_p_product_orders']).round(5)
    
    df_all['unknown_buyer'] = np.where(df_all['customer_id'] == -1, True, False)
    df_all['description_upppercase'] = df_all['description'].str.isupper().fillna(False)
    #df_all['country_aggregated'] = df_all['country'].apply(lambda x: 'Other' if any([country in x for country in countries]) else x)
    #df_all['cat_country'] = pd.factorize(df_all['country'])[0]
    #df_all['cat_country2'] = pd.factorize(df_all['country_aggregated'])[0]
    #df_all['invalid_transaction'] = 
    df_all['log_price_total'] = np.log2(df_all['price_total'] + 6).round(2)
    df_all = get_invoice_date_parameters(df_all)
    #df_all = get_additional_bool_column_from_description(df_all)
    return df_all

def prepare_additional_features(df):
    def group_to_dict(group_key, column, agg_func=np.sum):
        dict_ = df.groupby(group_key)[column].agg(agg_func).to_dict()
        if -1 in dict_: 
            del dict_[-1]
        mean = np.mean( list(dict_.values()) )
        return defaultdict(lambda: mean, dict_)
    df['different_items'] = df['invoice'].map(group_to_dict('invoice','stock_code', agg_func=np.size))
    #df['all_quantity'] = df['invoice'].map(group_to_dict('invoice','quantity', agg_func=np.sum))
    df['price_unit_median'] = df['invoice'].map(group_to_dict('invoice', 'price_unit', agg_func=np.median))
    df['log_price_full_invoice'] = np.log2(df['invoice'].map(group_to_dict('invoice', 'price_total', agg_func=np.sum)) + 6)
    df['max_return_product_invoice'] = df['invoice'].map(group_to_dict('invoice', 'cnt_p_product_cancel', agg_func=np.max))
    df['ratio_p_product_orders'] = df['invoice'].map(group_to_dict('invoice', 'ratio_p_product_orders', agg_func=np.max))
    return df
        