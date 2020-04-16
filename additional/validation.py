import numpy as np

def cross_validation(train_h5, data_perc=0.1, test_perc=0.3, k_folds=3):
    np.random.seed(0)
    def get_invoices(df, quarter, to_remove=None):
        if to_remove:
            return list(np.random.choice(list(df[~df['invoice'].isin(to_remove)].invoice), min(quarter, len(df[~df['invoice'].isin(to_remove)])), replace=False))
        return list(np.random.choice(list(df.invoice), min(quarter, len(df)), replace=False))
    train = train_h5.groupby('invoice').agg(price_total=('price_total', 'sum'), is_canceled=('is_canceled', 'first'))
    canceled = train[train['is_canceled'] == True].reset_index()
    not_canceled = train[train['is_canceled'] == False].reset_index()
    l_c = len(canceled)
    l_nc = len(not_canceled)
    full_len = l_c + l_nc
    multiplier = l_nc / l_c
    light_data = int(full_len * data_perc)
    test_data = int(light_data * test_perc)
    train_data = light_data - test_data

    number_of_canceled = int(test_data / (multiplier+1))
    number_of_not_canceled = test_data - number_of_canceled
    quarter_canceled = int(number_of_canceled/4)
    quarter_not_canceled = int(number_of_not_canceled/4)
    
    number_of_canceled_t = int(train_data / (multiplier+1))
    number_of_not_canceled_t = train_data - number_of_canceled_t
    quarter_canceled_t = int(number_of_canceled_t/4)
    quarter_not_canceled_t = int(number_of_not_canceled_t/4)
    
    nc_25, nc_50, nc_75 = not_canceled.describe().iloc[4][1], not_canceled.describe().iloc[5][1], not_canceled.describe().iloc[6][1]
    c_25, c_50, c_75 = canceled.describe().iloc[4][1], canceled.describe().iloc[5][1], canceled.describe().iloc[6][1]
    not_canceled_1 = not_canceled[not_canceled['price_total'] <= nc_25]
    not_canceled_2 = not_canceled[(not_canceled['price_total'] > nc_25) & (not_canceled['price_total'] <= nc_50)]
    not_canceled_3 = not_canceled[(not_canceled['price_total'] > nc_50) & (not_canceled['price_total'] <= nc_75)]
    not_canceled_4 = not_canceled[not_canceled['price_total'] > nc_75]
    canceled_1 = canceled[canceled['price_total'] <= c_25]
    canceled_2 = canceled[(canceled['price_total'] > c_25) & (canceled['price_total'] <= c_50)]
    canceled_3 = canceled[(canceled['price_total'] > c_50) & (canceled['price_total'] <= c_75)]
    canceled_4 = canceled[canceled['price_total'] > c_75]
    
    for _ in range(k_folds):
        nc_1 = get_invoices(not_canceled_1, quarter_not_canceled)
        nc_2 = get_invoices(not_canceled_2, quarter_not_canceled)
        nc_3 = get_invoices(not_canceled_3, quarter_not_canceled)
        nc_4 = get_invoices(not_canceled_4, quarter_not_canceled)
        c_1 = get_invoices(canceled_1, quarter_canceled)
        c_2 = get_invoices(canceled_2, quarter_canceled)
        c_3 = get_invoices(canceled_3, quarter_canceled)
        c_4 = get_invoices(canceled_4, quarter_canceled)
        test_invoices = nc_1 + nc_2 + nc_3 + nc_4 + c_1 + c_2 + c_3 + c_4
        test_idx = train_h5[train_h5['invoice'].isin(test_invoices)].index
        nct_1 = get_invoices(not_canceled_1, quarter_not_canceled_t, nc_1)
        nct_2 = get_invoices(not_canceled_2, quarter_not_canceled_t, nc_2)
        nct_3 = get_invoices(not_canceled_3, quarter_not_canceled_t, nc_3)
        nct_4 = get_invoices(not_canceled_4, quarter_not_canceled_t, nc_4)
        ct_1 = get_invoices(canceled_1, quarter_canceled_t, c_1)
        ct_2 = get_invoices(canceled_2, quarter_canceled_t, c_2)
        ct_3 = get_invoices(canceled_3, quarter_canceled_t, c_3)
        ct_4 = get_invoices(canceled_4, quarter_canceled_t, c_4)
        train_invoices = nct_1 + nct_2 + nct_3 + nct_4 + ct_1 + ct_2 + ct_3 + ct_4
        train_idx = train_h5[train_h5['invoice'].isin(train_invoices)].index
        yield train_idx, test_idx
