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



