# Produce Destination Freq plot

'''

    s = raw_data.Destination
    counts = s.value_counts()
    percent = s.value_counts(normalize=True)
    fig, ax = plt.subplots()
    percent100 = s.value_counts(normalize=True).mul(100).round(1).cumsum().plot(ax=ax, kind='line')
    plt.show()

'''

###================================================================================###
###================================================================================###
###================================================================================###



        # acc = score.accuracy(y_test, y_hat)
        # plt.title("Difference")
        # plt.scatter(np.arange(len(y_test)), y_test, label='test', color='r' )
        # plt.scatter(np.arange(len(y_hat)), y_hat, label='prediction', color='b')
        # plt.savefig('plots/RandomForest/scatter.pdf')
        # plt.legend()
        # plt.show()

        


"""
def complete_missing_columns_new(X_predict, X_train_df):
        base_columns = list(X_train_df.columns)
        new_columns = list(X_predict.columns)
        n_new, _ = X_predict.shape
        zero_vec = np.zeros((n_new, 1))
        X_new = X_predict.values
        new_data_matrix = np.zeros((n_new, X_train_df.shape[1]))
        new_df = pd.DataFrame(data=new_data_matrix, columns=base_columns)
        for i in range(len(base_columns)):
            if base_columns[i] in new_columns:
                indx = new_columns.index(base_columns[i])
                new_df[base_columns[i]] = X_new[:, indx]
            else:
                new_df[base_columns[i]] = zero_vec
        return new_df
def target_encode_column_dict_df(X, enc_map, column):
    X[column] = X[column].apply(lambda x: enc_map.get(x))
    return X
march2022_path = 'Future Schedule 20220301-20220331.xlsx'
march2022 = pd.read_excel(march2022_path)

pred_data = transformer.transform(march2022)
pred_data.drop(['FlightNumber', "Month", 'Day',
               'Sector', 'Year'], axis=1, inplace=True)

test = complete_missing_columns_new(pred_data, X_df)
test1 = target_encode_column_dict_df(X=test, enc_map=enc_map2, column='Destination')
dest = test1.Destination

"""
#df.loc[df['favorite_color'].isin(array)]


