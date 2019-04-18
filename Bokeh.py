### Interactive Data Visualization with Bokeh

### 1. Basic plotting with Bokeh

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Import output_file and show from bokeh.io
from bokeh.io import output_file, show

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility, female_literacy)

# Call the output_file() function and specify the name of the file
output_file('fert_lit.html')

# Display the plot
show(p)

# Create the figure: p
p = figure(x_axis_label='fertility', y_axis_label='female_literacy (% population)')

# Add a circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica)

# Add an x glyph to the figure p
p.x(fertility_africa, female_literacy_africa)

# Specify the name of the file
output_file('fert_lit_separate.html')

# Display the plot
show(p)

# Create the figure: p
p = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a blue circle glyph to the figure p
p.circle(fertility_latinamerica, female_literacy_latinamerica, color = "blue", size=10, alpha=0.8)

# Add a red circle glyph to the figure p
p.circle(fertility_africa, female_literacy_africa, color = "red", size=10, alpha=0.8)

# Specify the name of the file
output_file('fert_lit_separate_colors.html')

# Display the plot
show(p)

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type="datetime": p
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x axis and price along the y axis
p.line(date, price)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)

# Import figure from bokeh.plotting
from bokeh.plotting import figure

# Create a figure with x_axis_type='datetime': p
p = figure(x_axis_type='datetime', x_axis_label='Date', y_axis_label='US Dollars')

# Plot date along the x-axis and price along the y-axis
p.line(date, price)

# With date on the x-axis and price on the y-axis, add a white circle glyph of size 4
p.circle(date, price, fill_color='white', size=4)

# Specify the name of the output file and show the result
output_file('line.html')
show(p)

# Create a list of az_lons, co_lons, nm_lons and ut_lons: x
x = [az_lons, co_lons, nm_lons, ut_lons]

# Create a list of az_lats, co_lats, nm_lats and ut_lats: y
y = [az_lats, co_lats, nm_lats, ut_lats]

# Add patches to figure p with line_color=white for x and y
p.patches(x, y, line_color='white')

# Specify the name of the output file and show the result
output_file('four_corners.html')
show(p)

# Import numpy as np
import numpy as np

# Create array using np.linspace: x
x = np.linspace(0, 5, 100)

# Create array using np.cos: y
y = np.cos(x)

# Add circles at x and y
p.circle(x, y)

# Specify the name of the output file and show the result
output_file('numpy.html')
show(p)

# Import pandas as pd
import pandas as pd

# Read in the CSV file: df
df = pd.read_csv('auto.csv')

# Import figure from bokeh.plotting
from bokeh.plotting import figure 

# Create the figure: p
p = figure(x_axis_label='HP', y_axis_label='MPG')

# Plot mpg vs hp by color
p.circle(df.hp, df.mpg, color = df.color, size = 10)

# Specify the name of the output file and show the result
output_file('auto-df.html')
show(p)

# Import the ColumnDataSource class from bokeh.plotting
from bokeh.plotting import ColumnDataSource

# Create a ColumnDataSource from df: source
source = ColumnDataSource(df)

# Add circle glyphs to the figure p
p.circle('Year', 'Time', source = source, color = 'color', size = 8)

# Specify the name of the output file and show the result
output_file('sprint.html')
show(p)

# Create a figure with the "box_select" tool: p
p = figure(x_axis_label = 'Year', y_axis_label = 'Time', tools = 'box_select')

# Add circle glyphs to the figure p with the selected and non-selected properties
p.circle('Year', 'Time', source = source, selection_color = 'red', nonselection_alpha = 0.1)

# Specify the name of the output file and show the result
output_file('selection_glyph.html')
show(p)

# import the HoverTool
from bokeh.models import HoverTool

# Add circle glyphs to figure p
p.circle(x, y, size=10,
         fill_color='grey', alpha=0.1, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')

# Create a HoverTool: hover
hover = HoverTool(tooltips=None, mode='vline')

# Add the hover tool to the figure p
p.add_tools(hover)

# Specify the name of the output file and show the result
output_file('hover_glyph.html')
show(p)

#Import CategoricalColorMapper from bokeh.models
from bokeh.models import CategoricalColorMapper

# Convert df to a ColumnDataSource: source
source = ColumnDataSource(df)

# Make a CategoricalColorMapper object: color_mapper
color_mapper = CategoricalColorMapper(factors=['Europe', 'Asia', 'US'],
                                      palette=['red', 'green', 'blue'])

# Add a circle glyph to the figure p
p.circle('weight', 'mpg', source=source,
            color=dict(field='origin', transform=color_mapper),
            legend='origin')

# Specify the name of the output file and show the result
output_file('colormap.html')
show(p)

#######################

### 2. Layouts, Interactions, and Annotations

# Import row from bokeh.layouts
from bokeh.layouts import row

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p1
p1.circle('fertility', 'female_literacy', source = source)

# Create the second figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')

# Add a circle glyph to p2
p2.circle('population', 'female_literacy', source = source)

# Put p1 and p2 into a horizontal row: layout
layout = row(p1, p2)

# Specify the name of the output_file and show the result
output_file('fert_row.html')
show(layout)

# Import column from the bokeh.layouts module
from bokeh.layouts import column

# Create a blank figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female_literacy (% population)')

# Add circle scatter to the figure p1
p1.circle('fertility', 'female_literacy', source=source)

# Create a new blank figure: p2
p2 = figure(x_axis_label='population', y_axis_label='female_literacy (% population)')

# Add circle scatter to the figure p2
p2.circle('population', 'female_literacy', source = source)

# Put plots p1 and p2 in a column: layout
layout = column(p1, p2)

# Specify the name of the output_file and show the result
output_file('fert_column.html')
show(layout)

# Import column and row from bokeh.layouts
from bokeh.layouts import row, column

# Make a column layout that will be used as the second row: row2
row2 = column([mpg_hp, mpg_weight], sizing_mode='scale_width')

# Make a row layout that includes the above column layout: layout
layout = row([avg_mpg, row2], sizing_mode='scale_width')

# Specify the name of the output_file and show the result
output_file('layout_custom.html')
show(layout)

# Import gridplot from bokeh.layouts
from bokeh.layouts import gridplot

# Create a list containing plots p1 and p2: row1
row1 = [p1, p2]

# Create a list containing plots p3 and p4: row2
row2 = [p3, p4]

# Create a gridplot using row1 and row2: layout
layout = gridplot([row1, row2])

# Specify the name of the output_file and show the result
output_file('grid.html')
show(layout)

# Import Panel from bokeh.models.widgets
from bokeh.models.widgets import Panel

# Create tab1 from plot p1: tab1
tab1 = Panel(child=p1, title='Latin America')

# Create tab2 from plot p2: tab2
tab2 = Panel(child=p2, title='Africa')

# Create tab3 from plot p3: tab3
tab3 = Panel(child=p3, title='Asia')

# Create tab4 from plot p4: tab4
tab4 = Panel(child=p4, title='Europe')

# Import Tabs from bokeh.models.widgets
from bokeh.models.widgets import Tabs

# Create a Tabs layout: layout
layout = Tabs(tabs=[tab1, tab2, tab3, tab4])

# Specify the name of the output_file and show the result
output_file('tabs.html')
show(layout)

# Link the x_range of p2 to p1: p2.x_range
p2.x_range = p1.x_range

# Link the y_range of p2 to p1: p2.y_range
p2.y_range = p1.y_range

# Link the x_range of p3 to p1: p3.x_range
p3.x_range = p1.x_range

# Link the y_range of p4 to p1: p4.y_range
p4.y_range = p1.y_range

# Specify the name of the output_file and show the result
output_file('linked_range.html')
show(layout)

# Create ColumnDataSource: source
source = ColumnDataSource(data)

# Create the first figure: p1
p1 = figure(x_axis_label='fertility (children per woman)', y_axis_label='female literacy (% population)', tools='box_select,lasso_select')

# Add a circle glyph to p1
p1.circle('fertility', 'female literacy', source=source)

# Create the second figure: p2
p2 = figure(x_axis_label='fertility (children per woman)', y_axis_label='population (millions)', tools='box_select,lasso_select')

# Add a circle glyph to p2
p2.circle('fertility', 'population', source=source)

# Create row layout of figures p1 and p2: layout
layout = row([p1, p2])

# Specify the name of the output_file and show the result
output_file('linked_brush.html')
show(layout)

# Add the first circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=latin_america, size=10, color='red', legend='Latin America')

# Add the second circle glyph to the figure p
p.circle('fertility', 'female_literacy', source=africa, size=10, color='blue', legend='Africa')

# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)

# Assign the legend to the bottom left: p.legend.location
p.legend.location = 'bottom_left'

# Fill the legend background with the color 'lightgray': p.legend.background_fill_color
p.legend.background_fill_color = 'lightgray'

# Specify the name of the output_file and show the result
output_file('fert_lit_groups.html')
show(p)

# Import HoverTool from bokeh.models
from bokeh.models import HoverTool

# Create a HoverTool object: hover
hover = HoverTool(tooltips = [('Country','@Country')])

# Add the HoverTool object to figure p
p.add_tools(hover)

# Specify the name of the output_file and show the result
output_file('hover.html')
show(p)

#######################

### 3. Building interactive apps with Bokeh




























