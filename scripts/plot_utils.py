import numpy
import matplotlib.pyplot as plt

datasets_crossectional = ['activatedsludge', 'Glacier', 'River', 'Environmental Terrestrial Soil', 'ORAL', 'Environmental Aquatic Marine', 'Lake', 'GUT']

environment_cmap_dict = {'Environmental Terrestrial Soil':'forestgreen', 'Glacier':'paleturquoise', 'River':'darkorchid', 'activatedsludge':'saddlebrown', 'ORAL':'gold', 'Environmental Aquatic Marine':'dodgerblue', 'Lake':'midnightblue', 'GUT':'firebrick'}
environment_shape_dict = {'Environmental Terrestrial Soil':'x', 'Glacier':'^', 'River':'D', 'activatedsludge':'o', 'ORAL':'h', 'Environmental Aquatic Marine':'v', 'Lake':'+', 'GUT':'s'}
environment_facecolor_dict = {'activatedsludge':'none', 'Glacier':'none', 'River':'none', 'Environmental Terrestrial Soil':environment_cmap_dict['Environmental Terrestrial Soil'], 'ORAL':'none', 'Environmental Aquatic Marine':'none', 'Lake':environment_cmap_dict['Lake'], 'GUT':'none'}
# edgecolot is just cmap
environment_name_dict = {'oralcavity': 'Human oral', 'skin': 'Human skin', 'feces': 'Human gut', 'activatedsludge': 'Sludge', 'Glacier':'Glacier', 'River':'River', 'Environmental Terrestrial Soil':'Soil', 'ORAL':'Human oral', 'seawater':'Marine', 'Environmental Aquatic Marine':'Marine', 'VAGINAL':'Vaginal', 'Lake':'Lake', 'GUT':'Human gut'}

sname_to_environment_dict = {'lake':'Lake', 'seawater':'Environmental Aquatic Marine', 'glacier':'Glacier', 'gut1':'GUT', 'oral1':'ORAL', 'sludge':'activatedsludge', 'river':'River', 'soil':'Environmental Terrestrial Soil'}



# for example plots
id_val_example = 'SRP056641 GUT'
environment_example = 'GUT'
sname_example = 'gut1'

c_blue='#1E90FF'
c_orange='#EB5900'
tick_labelsize=8
size_x, size_y = 4,4
lw=3
scatter_size=40


def count_pts_within_radius(x, y, radius, logscale=0):
    """Count the number of points within a fixed radius in 2D space"""
    #todo: see if we can improve performance using KDTree.query_ball_point
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html
    #instead of doing the subset based on the circle
    unique_points = set([(x[i], y[i]) for i in range(len(x))])
    count_data = []
    logx, logy, logr = numpy.log10(x), numpy.log10(y), numpy.log10(radius)
    for a, b in unique_points:
        if logscale == 1:
            loga, logb = numpy.log10(a), numpy.log10(b)
            num_neighbors = len(x[((logx - loga) ** 2 +
                                   (logy - logb) ** 2) <= logr ** 2])
        else:
            num_neighbors = len(x[((x - a) ** 2 + (y - b) ** 2) <= radius ** 2])
        count_data.append((a, b, num_neighbors))
    return count_data


def plot_color_by_pt_dens(x, y, radius, loglog=0, plot_obj=None, cmap='Blues', norm=None, alpha=1, size=8):
    """Plot bivariate relationships with large n using color for point density

    Inputs:
    x & y -- variables to be plotted
    radius -- the linear distance within which to count points as neighbors
    loglog -- a flag to indicate the use of a loglog plot (loglog = 1)

    The color of each point in the plot is determined by the logarithm (base 10)
    of the number of points that occur with a given radius of the focal point,
    with hotter colors indicating more points. The number of neighboring points
    is determined in linear space regardless of whether a loglog plot is
    presented.
    """
    plot_data = count_pts_within_radius(x, y, radius, loglog)
    sorted_plot_data = numpy.array(sorted(plot_data, key=lambda point: point[2]))
    #color_range = np.sqrt(sorted_plot_data[:, 2])
    #color_range = color_range / max(color_range)

    #colors = [ cm.YlOrRd(x) for x in color_range ]

    if plot_obj == None:
        plot_obj = plt.axes()

    if loglog == 1:
        plot_obj.set_xscale('log')
        plot_obj.set_yscale('log')
        cvals = numpy.sqrt(sorted_plot_data[:, 2])

        plot_obj.scatter(sorted_plot_data[:, 0], sorted_plot_data[:, 1],s=size,
                         c=cvals, cmap=cmap, norm=norm, edgecolors='none', alpha=alpha)
        plot_obj.set_xlim(min(x) * 0.5, max(x) * 2)
        plot_obj.set_ylim(min(y) * 0.5, max(y) * 2)
    else:
        cvals = numpy.log10(sorted_plot_data[:, 2])
        plot_obj.scatter(sorted_plot_data[:, 0], sorted_plot_data[:, 1],s=size,
                    c=cvals, cmap=cmap, norm=norm, edgecolors='none', alpha=alpha)
    return plot_obj