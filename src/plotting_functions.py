

def plot_model_predictions(model_list, X_fwds_cv, filename=None):
    ''' takes the list of models and plots their forecasted change in rates
    This currently only handles three models (for the chart)
    INPUTS:
        model_list  a list of dictionaries with a 'name' key and a 'forecast' key that contains a 2D nparray
        X_fwds_cv   the cross validation data - this is used to extract column names and dates for the chart
        filename    if the file is passed a filename, it will save the file, othwewise it will show the plot

    '''
    import numpy as np
    import matplotlib.pyplot as plt
    # for now we have assumed that the plots will be have ylines +/- 2 bps
    y_max = 0.0002
    y_min = -0.0002
    x_tick_index = [1, 101, 201, 301, 401]

    #need to determine the rate titles
    column_names = X_fwds_cv.columns
    ind = np.arange(7,14)
    rate_titles = column_names[ind]

    # pull the dates from the X_fwds_cv dataframe
    these_dates = X_fwds_cv.index

    fig, axs = plt.subplots(7, 3, figsize = (15,15))
    fig.suptitle('Initial Model Predictions', fontsize=24)

    for i, ax in enumerate(axs.flatten()):

        this_row = int(i/3)
        this_rate = rate_titles[this_row]

        this_column = i%3
        this_name = model_list[this_column]['name']
        this_title = this_name + ' forecast for ' + this_rate + ' forward'
        ax.plot(these_dates, model_list[this_column]['forecast'][:,this_row])
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 250))
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(4)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        ax.set_title(this_title, fontsize=8)
        ax.set_ylim(bottom=y_min, top=y_max)
        ax.grid()
        ax.set_facecolor('whitesmoke')
    plt.tight_layout()
    fig.subplots_adjust(top = .92)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


