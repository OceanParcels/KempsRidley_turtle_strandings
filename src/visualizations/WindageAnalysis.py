"""
Use input from the ExtractColdStunningEvent.py run where each particle's outputs were stored.
This code produces violin plots for days before stranding and distance from stranding location.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

home_folder = '/Users/dmanral/Desktop/Analysis/Ridley/RESULTS_27_03_2023/'
winds = ['0pWind', '01pWind', '1pWind', '2pWind', '3pWind']
windage = ['0.0%', '0.1%', '1.0%', '2.0%', '3.0%']

# output format->
# full_data[st, indices, 0, tc] = lats
# full_data[st, indices, 1, tc] = lons
# full_data[st, indices, 2, tc] = days[indices]
# full_data[st, indices, 3, tc] = dist

fig = make_subplots(rows=2, cols=3, subplot_titles=('10°C', '12°C', '14°C'),
                    shared_xaxes=True,
                    shared_yaxes=True,
                    vertical_spacing=0.1)
set_legend = True
for tc_index, tc in enumerate(['10C', '12C', '14C']):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, w in enumerate(winds):
        full_ds = np.load(home_folder + 'full_data_{0}.npy'.format(winds[i]))
        ds_days = full_ds[:, :, 2, tc_index].flatten()
        ds_dist = full_ds[:, :, 3, tc_index].flatten()
        fig.add_trace(go.Violin(y=ds_days,
                                name=w,
                                box_visible=True,
                                meanline_visible=True,
                                marker=dict(color=colors[i]),
                                showlegend=set_legend), row=1, col=tc_index + 1)
        fig.data[-1].update(name=windage[i], legendgroup=windage[0])

        fig.add_annotation(x=i, y=75,
                           text='{0}'.format(np.round(np.count_nonzero(~np.isnan(ds_days)) / len(ds_days), 2)),
                           showarrow=False,
                           xref='x{0}'.format(tc_index + 1),
                           yref='y1',
                           yshift=0,
                           font_size=17,
                           bgcolor='gainsboro',
                           borderpad=2)
        fig.add_trace(go.Violin(y=ds_dist,
                                name=w,
                                box_visible=True,
                                meanline_visible=True,
                                marker=dict(color=colors[i]),
                                showlegend=False), row=2, col=tc_index + 1)
        fig.data[-1].update(name=windage[i], legendgroup=windage[0])
    set_legend = False
fig.update_yaxes(title_text="Days before stranding", row=1, col=1)
fig.update_yaxes(title_text="Distance from stranding location (km)", row=2, col=1)
fig.update_xaxes(title_text="Windage % settings", row=2, col=2)
fig.update_layout(height=1000, width=1200,
                  title_text="Cold stunning event based on temperature thresholds for different Windage settings",
                  font_family="sans-serif",
                  template='simple_white',
                  xaxis1_showticklabels=True,
                  xaxis2_showticklabels=True,
                  xaxis3_showticklabels=True,
                  yaxis2_showticklabels=True,
                  yaxis3_showticklabels=True,
                  yaxis5_showticklabels=True,
                  yaxis6_showticklabels=True,
                  title_font_size=20,
                  font_size=18,
                  legend_font_size=20)
fig.show()
fig.write_image(home_folder + 'windage_analysis.jpeg', scale=5)
