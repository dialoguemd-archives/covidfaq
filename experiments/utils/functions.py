import os
import pickle
from datetime import datetime
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import euclidean_distances as pariwise_euclidean_distance
import math


# Arguments
def print_config(args,header="CONFIGURATION", skip_list=[]):
    # @todo - make header configurable
    print("\n\n---------------------------- CONFIGURATION --------------------------------\n")
    arguments = vars(args)
    for k, v in arguments.items():
        if k not in skip_list:
            print("\x1b[36m{}\x1b[0m : {}".format(k,v))
    print("\n----------------------------------------------------------------------------\n")

def equal(*vals):
    if not vals: raise ValueError("No values to compare.")
    first = vals[0]
    return all(first == rest for rest in vals)

def file_exists(fpath):
    """
        Checks if a file exists, legitimately, not just in name
    """
    # @todo - Use any()
        
    if os.path.exists(fpath) and os.path.isfile(fpath) and os.stat(fpath).st_size != 0:
        return True
    else: 
        return False

def quick_dump(what, where, **args):
    """
        Just say what and where and leave the rest to me
    """
    # @todo - Add couple of checks etc, isDir, extension, isFile, exists? overwrite?

    pickle.dump(what,open(where,"wb"),**args)


# @todo - if want to save dict as json
def save_dict(dict_to_save,loc):
    # Start pickling
    if ".json" not in loc:
        loc = loc + ".json"
    json.dump(dict_to_save, open(loc, 'w+'), indent=4)


def save_dict_as_csv(dict_to_save,loc):
    # To save a dictionary as a csv file, where each key is a header, even if nested
    with open(loc, 'w+', newline='') as csvfile:
        fieldnames = []
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for key, val in dict_to_save.items():
            if type(val) == dict:
                # Work in Progress - need recursion
                return


def sequential_dump(list_of_whats, where, prefix, **args):
    """
        Just say what and where and leave the rest to me
    """
    # @todo - pad the number in filename
    for i, what in enumerate(list_of_whats,1):
        here = os.path.join(where,"{}_{}.p".format(prefix,i))
        pickle.dump(what,open(here,"wb"),**args)


# Plots
import matplotlib.pyplot as plt

def quick_plot(
                    x, y, 
                    subplot_labels=[], 
                    title=None, 
                    x_label="X", 
                    y_label="Y", 
                    legend_loc="lower right",
                    save_as=None, 
                    colors_dict={},
                    returnplt = False

                ):

    if type(x[0]) == list:
        plots = len(x)
    else:
        plots = 1
        x = [x]
        y = [y]

    if not colors_dict:
        colors_dict =   {
                            "slate blue"        :"#6266D6",
                            "medium turquoise"  :"#3DD3CC",
                            "fuchsia"           :"#BC47C6",
                            "hot pink"          :"#FF72B8",
                            "sunny"             :"#F7F076",
                            "pastel orange"     :"#FFBA49",
                            "teal green"        :"#048A81",
                            "red orange"        :"#F95252",
                            "oxford blue"       :"#031D44",
                            "mardi gras"        :"#8A1C7C",
                            "rebecca purple"    :"#6E2594",
                            "dollar bill"       :"#93BA55"
                        }
        #colors = ["#F652A0","#7B56D4","#04ECF0","#07DAC0"]
        colors = list(colors_dict.values())
        colors = [
                    "#BC47C6", "#FF72B8", "#fdf574", "#FFBA49",
                    "#0fd4d4", "#847cff", "#43e6f7", "#3de680",
                    "#9ef572", "#fbe64e", "#ff6c04", "#ff3535"
                 ]

    if len(subplot_labels) == 0:
        subplot_labels = [f"Plot{i}" for i in range(1,plots+1)]\

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot Design
    ax.set_facecolor("#ffffff")
    for pos in ["bottom","top","right","left"]:
        ax.spines[pos].set_color('#dddddd')

    ax.tick_params(axis='x', colors='#444444')
    ax.tick_params(axis='y', colors='#444444')

    ax.yaxis.label.set_color('#444444')
    ax.xaxis.label.set_color('#444444')

    font_dict={
                "fontweight":"light",
                "fontname"  :"Courier",
                "color"     :"#444444",
                "fontsize"  :12,
                "wrap"      :True
              }

    # Actual Plotting
    for x_vals, y_vals, label,color in zip(x, y, subplot_labels, colors[:plots]):    
        ax.plot(list(x_vals), list(y_vals), color, linestyle="-", label=label)

    if plots > 1: plt.legend(loc=legend_loc)

    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)

    if not title: title = "Plot"
    ax.set_title(title,fontdict=font_dict, pad=10)

    # plt.clf()

    if returnplt: return plt, ax
    else:         
        if save_as: plt.savefig(save_as)
        plt.show()

def get_ax():

    plt.clf()
    ax = plt.gca()

    # Plot Design
    ax.set_facecolor("#ffffff")
    for pos in ["bottom","top","right","left"]:
        ax.spines[pos].set_color('#dddddd')

    ax.tick_params(axis='x', colors='#444444')
    ax.tick_params(axis='y', colors='#444444')

    ax.yaxis.label.set_color('#444444')
    ax.xaxis.label.set_color('#444444')

    # font_dict={
    #             "fontweight":"light",
    #             "fontname"  :"Courier",
    #             "color"     :"#444444",
    #             "fontsize"  :12,
    #             "wrap"      :True
    #           }

    # ax.set_xlabel(labelpad=10)
    # ax.set_ylabel(labelpad=10)

    # ax.set_title(fontdict=font_dict, pad=10)

    return ax


# Very specific to pandas dataframe
def add_to_col_as_list(dataframe, index, column_name, value):
    # If the entity label is not in the dataframe as a colum, create it
    if column_name not in dataframe.columns:
        dataframe.at[index,column_name] = [[value]]
    else:
    # If all is good, append to the list. That is column exists, and we have an existing list to append to
        if type(dataframe.loc[index,column_name]) != list:
            dataframe.loc[index,column_name] = [[value]]
        else:
            dataframe.loc[index,column_name].append(value)    

# Flatten list
def flatten(l,unique=False):
    flattened_list = []
    for item in l:
        if type(item) is str:
            flattened_list.append(item)
            continue
        try:   
            if iter(item):
                flattened_list.extend(flatten(item))
        except:
            flattened_list.append(item)
    if unique:
        return list(set(flattened_list))     
    else:        
        return flattened_list


def to_date(d, date_format = "%Y-%m-%d %H:%M:%S.%f"):
    """
    To do - Accept Numpy arrays, tuples etc
    """
    if type(d) == pd.core.series.Series:
        d = list(d)
    if type(d) == list:
        return [datetime.strptime(date,date_format) if type(date) == str else date for date in d]
    elif type(d) == str:
        return datetime.strptime(d,date_format)
    else:
        raise ValueError("Either String or list of Strings is accepted.")
        
def rbf(args,gamma = None):
    """
    Calculates RBF valus
    """
    if type(args) == pd.core.series.Series:
        args = list(args)
        
    if type(args) == list:
        return [curve(vals, gamma) for vals in args]
    else:
        if gamma is None:
            gamma = 1/args.days_threshold

        # I don't know the name of the function - credits Parsa - RBF Kernel - Radial Basis Function
        return math.exp(-1 * gamma * (args**2))

# Define functions
def header(name):
    '''
        Just to format output - Dashed underline to the title
    '''
    print_me = "\n" + name + "\n"
    print_me+= "-" * len(name)
    if return_val:
        return print_me