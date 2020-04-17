%%writefile custom_tools.py

import numpy as np
import pandas as pd
from collections import defaultdict 

dict_additional_columns = {
    'color_RED': ['RED'],
    'color_PINK': ['PINK', 'FUSHIA'],
    'color_BLUE': ['BLUE'],
    'color_GREEN': ['GREEN'],
    'color_WHITE': ['WHITE', 'WHIT'],
    'color_BLACK': ['BLACK', 'BLK', 'BLCK'],
    'color_CREAM': ['CREAM'],
    'color_GOLD': ['GOLD'],
    'color_SILVER': ['SILVER'],
    'color_COOPER': ['COOPER'],
    'color_ASSORTED': ['ASSORTED'],
    'cat_POSTAGE': ['POSTAGE'],
    'cat_SAMPLES': ['SAMPLES'],
    'cat_MANUAL': ['Manual'],
    'cat_FEES': ['Bank Charges', 'bank charges', 'AMAZON FEE', 'CRUK Commission'],
    'cat_ADJUSTMENT': ['Adjustment'],
    'cat_SET': ['SET'],
#     'cat_FURNITURE': ['DRAWER', 'CABINET', 'DRESSER', 'SEAT', 'SIDEBOARD', 'MIRROR', 'TABLE'],
    'cat_MATERIAL': ['QUILT', 'FLAG'],
    'material_WOOD': ['WOOD'],
    'material_CERAMIC': ['CERAMIC'],
    'material_GLASS': ['GLASS'],
    'material_GEM': ['AMETHYST', 'DIAMANTE', 'RUBY', 'AMBER', 'TURQUISE', 'QUARTZ', 'GEMSTONE', 'CRYSTAL', 'JADE'],
    'material_ENAMEL': ['ENAMEL'],
    'material_METAL': ['COOPER', 'ZINC', 'BRONZE'],
    'style_RETRO': ['RETRO'],
    'style_VINTAGE': ['VINTAGE'],
    'style_HISTORIC': ['EDWARDIAN', 'FRENCH', 'BAROQUE', 'MOROCCAN', 'ANTIQUE', 'ANT', 'RUSTIC', 'REGENCY'],
    'style_MODERN': ['MODERN', 'SCANDINAVIAN'],
    'type_JEWELRY': ['NECKLAGE', 'BEAD', 'RING', 'JEWEL', 'BRACELET'],
    'type_CHRISTMAS': ['CHRISTMAS'],
    'type_BAG': ['BAG'],
    'type_CONTAINER': ['TIN', 'BOX', 'CHEST', 'JAR']
}
dict_additional_columns.keys()

countries = ['Saudi Arabia', 'Czech Republic', 'Nigeria', 'Bermuda', 'West Indies', 'Lebanon', 'European Community',
           'Korea', 'Thailand', 'Brazil']

def prepare_train_and_test(train, test):
    test['is_canceled'] = np.NaN
    df_all = pd.concat([train, test], sort=False)
    # I didn't use those in the best result but I left them just in case you were curious what I was testing
    #df_all = prepare_product_customer_statistics(df_all)
    #df_all['country_aggregated'] = df_all['country'].apply(lambda x: 'Other' if any([country in x for country in countries]) else x)
    #df_all['cat_country'] = pd.factorize(df_all['country_aggregated'])[0]
    new_train_all_rows = df_all[~df_all['is_canceled'].isnull()].copy()
    new_test_all_rows = df_all[df_all['is_canceled'].isnull()].copy()
    new_train = prepare_dataframe(new_train_all_rows, train=True)
    new_test = prepare_dataframe(new_test_all_rows, train=False)
    return new_train, new_test

def prepare_product_customer_statistics(df_all):
    def group_to_dict(group_key, agg_func=np.sum):
        train = df_all[ ~df_all['is_canceled'].isnull()]
        dict_ = train.groupby(group_key)['is_canceled'].agg(agg_func).to_dict()
        if -1 in dict_: 
            del dict_[-1]
        mean = np.mean( list(dict_.values()) )
        return defaultdict(lambda: mean, dict_)
    df_all['cnt_p_product_cancel'] = df_all['stock_code'].map(group_to_dict('stock_code')).astype('float64')
    #df_all['cnt_p_product_cancel_country'] = df_all['stock_code'].map(group_to_dict(['stock_code', 'country'])).astype('float64')
    df_all['cnt_p_product_orders'] = df_all['stock_code'].map(group_to_dict('stock_code', agg_func=np.size))
    df_all['ratio_p_product_orders'] = (df_all['cnt_p_product_cancel']/df_all['cnt_p_product_orders']).round(5)
    df_all['cnt_customer_cancel'] = df_all['customer_id'].map(group_to_dict('customer_id')).astype('float64')
    df_all['cnt_customer_orders'] = df_all['customer_id'].map(group_to_dict('customer_id', agg_func=np.size))
    df_all['ratio_customer_orders'] = (df_all['cnt_customer_cancel']/df_all['cnt_customer_orders']).round(5)
    return df_all
    
def prepare_dataframe(df, train=False):
    prepared_df = create_orders_df(df, train)
    prepared_df = get_invoice_date_parameters(prepared_df)
    prepared_df = get_additional_bool_column_from_description(prepared_df, df)
    #prepared_df = one_hot_encoding(prepared_df, 'country')
    return prepared_df

def prepare_additional_features(df, rows_df):
    df_all['cnt_p_product_cancel'] = df_all['stock_code'].map(group_to_dict('stock_code')).astype('float64')
    #df_all['cnt_p_product_cancel_country'] = df_all['stock_code'].map(group_to_dict(['stock_code', 'country'])).astype('float64')
    df_all['cnt_p_product_orders'] = df_all['stock_code'].map(group_to_dict('stock_code', agg_func=np.size))
    df_all['ratio_p_product_orders'] = (df_all['cnt_p_product_cancel']/df_all['cnt_p_product_orders']).round(5)
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

def get_additional_column_from_description(prepared_df, df):
    columns = list(dict_additional_columns.keys())
    prepared_df[columns] = prepared_df.apply(lambda row: check_for_string(row.name, df, columns), axis=1)
    return prepared_df

def get_additional_bool_column_from_description(prepared_df, df):
    df['description'] = df['description'].astype(str)
    prepared_df['joined_descriptions'] = prepared_df.apply(lambda row: ' | '.join(df[df['invoice'] == row.name]['description']), axis=1)
    for column, words in dict_additional_columns.items():
        prepared_df[column] = prepared_df['joined_descriptions'].apply(lambda x: any([word in x for word in words]))
    return prepared_df

def create_orders_df(df, train=False):
    columns_count = ['quantity']
    columns_sum = ['price_total', 'quantity']
    columns_first = ['invoice_date', 'country_aggregated', 'customer_id', 'is_test']
    #, 'cnt_customer_cancel', 'cnt_customer_orders', 'ratio_customer_orders']
    columns_calculations = ['price_unit', 'price_total'] 
    #, 'cnt_p_product_cancel', 'cnt_p_product_orders', 'ratio_p_product_orders']
    if train:
        columns_sum.append('is_canceled')
    orders = df.groupby('invoice')[columns_sum].sum()
    if train:
        orders['is_canceled'] = orders['is_canceled'] > 0
        orders['total_return'] = orders['price_total'] * orders['is_canceled'] 
    orders_temp = df.groupby('invoice')[columns_first].first()
    orders = orders.join(orders_temp, how='inner')
    for column in columns_calculations:
        df[column] = df[column].astype(np.float32)
        grouped_single = df.groupby('invoice').agg({column: ['mean', 'min', 'max', 'median', 'std']})
        grouped_single.columns = [f'{column}_{calculation}' for calculation in ['mean', 'min', 'max', 'median', 'std']]
        grouped_single[f'{column}_std'] = grouped_single[f'{column}_std'].round(2).fillna(-1)
        if column != 'price_total':
            grouped_single[f'{column}_min_max_diff'] = grouped_single[f'{column}_max'] - grouped_single[f'{column}_min']
        orders = orders.join(grouped_single, how='inner')
    orders_temp = df.groupby('invoice')[columns_count].count()
    orders_temp = orders_temp.rename(columns={'quantity': 'different_items'})
    orders = orders.join(orders_temp, how='inner')
    orders['unknown_buyer'] = np.where(orders['customer_id'] == -1, True, False)
    orders['log_price_total'] = np.log2(orders['price_total'] + 6)
    return orders
    
def get_invoice_date_parameters(df):
    #df['year'] = df['invoice_date'].dt.year
    df['month'] = df['invoice_date'].dt.month
    #df['day'] = df['invoice_date'].dt.day
    df['hour'] = df['invoice_date'].dt.hour
    #df['minute'] = df['invoice_date'].dt.minute
    #df['day_of_year'] = df['invoice_date'].dt.dayofyear
    df['day_of_week'] = df['invoice_date'].dt.dayofweek
    #df['week_of_year'] = df['invoice_date'].dt.weekofyear
    #df['quarter'] = df['invoice_date'].dt.quarter
    df['weekend'] = np.where(df['day_of_week'] < 5, 0, 1)
    #df['parT_of_day'] = df['hour'].apply(lambda c: get_part_of_day(c))
    return df

def get_part_of_day(hour):
    return (
        0 if 5 <= hour <= 11
        else
        1 if 12 <= hour <= 17
        else
        2 if 18 <= hour <= 22
        else
        3
    )

def one_hot_encoding(df, column):
    df[column] = pd.Categorical(df[column])
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
    return df

def get_features(df, black_list=['total_return', 'is_canceled', 'is_test', 'price_unit_std', 'day_of_week']):
    feats = df.select_dtypes(include=[np.number, 'bool']).columns
    return [x for x in feats if x not in black_list]

def tidy_split(df, column, sep='|', keep=False):
    indexes = list()
    new_values = list()
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df
