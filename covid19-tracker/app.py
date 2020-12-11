#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:05:37 2020

@author: Menghong Han
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, MATCH
import time
import numpy as np
import plotly
from newsapi import NewsApiClient
import datetime
import pandas as pd
import pycountry
import json
import pandas as pd
import plotly.express as px
##################
#
# import streamlit as st
# import plotly.graph_objects as go
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# import dash_bootstrap_components as dbc
# from dash.dependencies import Input, Output, State
# import time
# #import torch
# #import nltk
# from typing import List
# import numpy as np
# #import json
# import plotly
# from newsapi import NewsApiClient
# from datetime import date, timedelta
# import urllib.request as requests
# import cv2
# from os import path, getcwd
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
# import cv2
# import random
# import wget
# import os
# import pandas as pd
# # import streamlit as st


# modules for generating the word cloud
# from os import path, getcwd
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
#
# def get_sources(country):
#     sources = newsapi.get_sources(country='us')
#     sources = [x['id'] for x in sources['sources']]
#     return sources
#
# def create_dataframe_last_30d(queries, sources):
#     fulldata = pd.DataFrame()
#     for q in queries:
#         for s in sources:
#             print(s)
#             json_data = newsapi.get_everything(q=q,
#                                               language='en',
#                                                 from_param=str(date.today() -timedelta(days=29)),
#                                               to= str(date.today()),
#                                                sources = s,
#                                               page_size=100,
#                                               page = 1,sort_by='relevancy'
#                                               )
#             if len(json_data['articles'])>0:
#                 data = pd.DataFrame(json_data['articles'])
#                 fulldata=pd.concat([fulldata,data])
#     if len(fulldata)>0:
#         fulldata['source'] = fulldata['source'].apply(lambda x : x['name'])
#         fulldata['publishedAt'] = pd.to_datetime(fulldata['publishedAt'])
#         fulldata = fulldata.drop_duplicates(subset='url').sort_values(by='publishedAt',ascending=False).reset_index()
#     return fulldata
#
# sources = get_sources(country='us')
# fulldf = create_dataframe_last_30d(['corona'],sources)
# fulldf.head()
#
# import plotly.express as px
# import textblob
#
# from textblob import TextBlob
# def textblob_sentiment(title,description):
#     blob = TextBlob(str(title)+" "+str(description))
#     return blob.sentiment.polarity
#
# fulldf['story_sentiment'] = fulldf.apply(lambda x: textblob_sentiment(x['title'],x['description']),axis=1)
#
# sent_df = fulldf.groupby('source').aggregate({'story_sentiment':np.mean,'index':'count'}).reset_index().sort_values('story_sentiment')
#
# relevant_sent_df = sent_df[sent_df['index']>10]
#
# px.bar(data_frame = relevant_sent_df,x = 'source',y='story_sentiment')
# def create_mask():
#     mask = np.array(Image.open("coronavirus.png"))
#     im_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#     _, mask = cv2.threshold(im_gray, thresh=20, maxval=255, type=cv2.THRESH_BINARY)
#     mask = 255 - mask
#     return mask
#
#
#
# mask = create_mask()
# def create_wc_by(source):
#     data = fulldf[fulldf['source']==source]
#     text = " ".join([x for x in data.content.values if x is not None])
#     stopwords = set(STOPWORDS)
#     stopwords.add('chars')
#     stopwords.add('coronavirus')
#     stopwords.add('corona')
#     stopwords.add('chars')
#     wc = WordCloud(background_color="white", max_words=1000, mask=mask, stopwords=stopwords,
#                max_font_size=90, random_state=42, contour_width=3, contour_color='steelblue')
#     wc.generate(text)
#     plt.figure(figsize=[50,50])
#     plt.imshow(wc, interpolation='bilinear')
#     plt.axis("off")
#     return plt
#
#
#
#   # Wordcloud
# mask = create_mask()
# sources = st.selectbox("NewsSource",relevant_sent_df.source.values, index=0)
# A= st.pyplot(create_wc_by(sources),use_container_width=True)
############################
# pd.options.mode.chained_assignment = None  # default='warn'



################################################################################
#
#    Data Manipulation
#
################################################################################

baseURL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"

def loadData(fileName, columnName):
    data = pd.read_csv(baseURL + fileName) \
        .melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
              var_name='Date', value_name=columnName)
    data['Province/State'].fillna('<all>', inplace=True)
    data[columnName].fillna(0, inplace=True)
    return data


allData = loadData(
    "time_series_covid19_confirmed_global.csv", "CumConfirmed") \
    .merge(loadData(
    "time_series_covid19_deaths_global.csv", "CumDeaths")) \
    .merge(loadData(
    "time_series_covid19_recovered_global.csv", "CumRecovered"))
allData['Date'] = pd.to_datetime(allData['Date'])
allData['CumRecovered'] = allData['CumRecovered'].fillna(0)
sum(allData['Province/State'] == 'MS Zaandam')

# Active Case = confirmed - deaths - recovered
allData['Active'] = allData['CumConfirmed'] - allData['CumDeaths'] - allData['CumRecovered']

full_grouped = allData.groupby(['Date', 'Country/Region'])[
    ['CumConfirmed', 'CumDeaths', 'CumRecovered', 'Active']].sum().reset_index()

# new cases
temp = full_grouped.groupby(['Country/Region', 'Date', ])[['CumConfirmed', 'CumDeaths', 'CumRecovered']]
temp = temp.sum().diff().reset_index()
mask = temp['Country/Region'] != temp['Country/Region'].shift(1)
temp.loc[mask, 'CumConfirmed'] = np.nan
temp.loc[mask, 'CumDeaths'] = np.nan
temp.loc[mask, 'CumRecovered'] = np.nan
# renaming columns
temp.columns = ['Country/Region', 'Date', 'New Cases', 'New Deaths', 'New Recovered']
# merging new values
full_grouped = pd.merge(full_grouped, temp, on=['Country/Region', 'Date'])
# filling na with 0
full_grouped = full_grouped.fillna(0)
# fixing data types
cols = ['New Cases', 'New Deaths', 'New Recovered']
full_grouped[cols] = full_grouped[cols].astype('int')
#
full_grouped['New Cases'] = full_grouped['New Cases'].apply(lambda x: 0 if x < 0 else x)
data_latest = full_grouped[full_grouped['Date'] == full_grouped['Date'].drop_duplicates().nlargest(1).iloc[-1]]
data_latest['Death_Rate'] = round(data_latest['CumDeaths'] / data_latest['CumConfirmed'] * 100)
data_latest['Recover_Rate'] = round(data_latest['CumRecovered'] / data_latest['CumConfirmed'] * 100)
country_match= allData[['Country/Region','Lat','Long']].drop_duplicates(subset = 'Country/Region')
data_latest =pd.merge(data_latest,country_match,how='left',on='Country/Region')


full_grouped['Death_Rate'] = round(full_grouped['CumDeaths'] / full_grouped['CumConfirmed'] * 100)
full_grouped['Recover_Rate'] = round(full_grouped['CumRecovered'] / full_grouped['CumConfirmed'] * 100)
country_match= allData[['Country/Region','Lat','Long']].drop_duplicates(subset = 'Country/Region')
full_grouped =pd.merge(full_grouped,country_match,how='left',on='Country/Region')


# add the ISO column

# list_countries = data_latest['Country/Region'].unique().tolist()
# # print(list_countries) # Uncomment to see list of countries
# d_country_code = {}  # To hold the country names and their ISO
# for country in list_countries:
#     try:
#         country_data = pycountry.countries.search_fuzzy(country)
#         # country_data is a list of objects of class pycountry.db.Country
#         # The first item  ie at index 0 of list is best fit
#         # object of class Country have an alpha_3 attribute
#         country_code = country_data[0].alpha_2
#         d_country_code.update({country: country_code})
#     except:
#         # print('could not add ISO 3 code for ->', country)
#         # If could not find country, make ISO code ' '
#         d_country_code.update({country: ' '})
# # create a new column iso_alpha in the df
# # and fill it with appropriate iso 3 code
# for k, v in d_country_code.items():
#     data_latest.loc[(data_latest['Country/Region'] == k), 'iso_alpha'] = v
#
# data_latest['iso_alpha'] = data_latest['iso_alpha'].apply(lambda x : str(x).lower())
#
# code_country_name_dict = data_latest.set_index('iso_alpha').to_dict()['Country/Region']
# country_name_code_dict = data_latest.set_index('Country/Region').to_dict()['iso_alpha']
# country_options = ['ae','ar','at','au','be','bg','br','ch','cn','co','cu','cz','de','eg','fr','gb','gr','hu','id','ie','il','in','it','jp','lt','lv','ma','mx','my','ng','nl','no','nz','ph','pl','pt','ro','rs','ru','sa','se','sg','si','sk','th','tr','ua','us','ve','za']
# country_options_st = [code_country_name_dict[x] for x in country_options]

################################################################################
#
#    Variable
#
################################################################################

graph_template = "plotly_dark"
ext_stylesheet = dbc.themes.DARKLY

criteria = {'CumConfirmed': 'Total Confirmed', 'CumDeaths': 'Total Deaths', 'CumRecovered': 'Total Recovered',
            'Active': 'Active', 'New Cases': 'New Cases', 'New Deaths': 'New Deaths', 'New Recovered': 'New Recovered'}
# ,'Death_Rate':'Death Rate','Recover_Rate':'Recover Rate'
# continents = ['All','Asia','Europe','Africa','North America','South America','Oceania']

cols = plotly.colors.DEFAULT_PLOTLY_COLORS




################################################################################
#
#    APP
#
################################################################################

app = dash.Dash('dash-covid19',
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=3"}],
                external_stylesheets=[ext_stylesheet])
app.title = "COVID-19 Tracker"

app_name = "COVID-19 Tracker"

server = app.server


################ TABS STYLING ####################

font_size = ".9vw"
color_active = "#F4F4F4"
color_inactive = "#AEAEAE"
color_bg = "#010914"

tabs_styles = {
    "flex-direction": "row",
}
tab_style = {
    "padding": "1.3vh",
    "color": color_inactive,
    "fontSize": font_size,
    "backgroundColor": color_bg,
}

tab_selected_style = {
    "fontSize": font_size,
    "color": color_active,
    "padding": "1.3vh",
    "backgroundColor": color_bg,
}


controls_1b = dbc.Form([
    dbc.FormGroup([
        dbc.Label(id='label_select_country', children=["Select the country"]),
        dcc.Dropdown(
            id='dropdown_country',
            options=[{'label': i, 'value': i} for i in data_latest['Country/Region'].unique()],
            value='US',
            style={'backgroundColor':'#F4F4F4', 'color':'black'}

        )
    ]
    )
])

controls_1c = dbc.Form([
    dbc.FormGroup([

        dbc.Label(id='label_select_criteria', children=["select the criteria"]),
        dcc.Dropdown(
            id='dropdown_criteria',
            options=[{'label': i, 'value': i} for i in criteria.keys()],
            value='CumConfirmed',
            style={'backgroundColor':'#F4F4F4', 'color':'black'}
        )
    ]
    )

])



################ TABS  ####################
feed_tabs =  dbc.Card([
        html.Div(
            dcc.Tabs(id="news_tab_right",
                     value="news-tab",
                     children=[
                         dcc.Tab(
                             label="Today's News Feed",
                             value="news-tab",
                             className="left-news-tab",
                             style=tab_style,
                             selected_style=tab_selected_style,
                         ),

                     ],
                     style=tabs_styles,
                     colors={
                         "border": None,
                         "primary": None,
                         "background": None,
                     },
                     ),
            className="left-tabs",
        ),
        dbc.CardBody(
            html.P(
                id="feed_content",
                className="left-col-feed-cards-text",
            ),
        ),
    ]
)
tab1 = dbc.Card([
    dbc.CardBody([
        # controls
        dbc.Row([
            # dbc.Col(controls_1a, md=4),
            dbc.Col(controls_1b, md=4),
            dbc.Col(controls_1c, md=4),

        ]),

        # info cards
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H5(id='label_info2', children=['Date'])),
                dbc.CardBody(
                         children =[html.Div("Last Updated: "),
                             html.H2(id='Date')
                             ,html.Br()
                                    ]
                             )
            ],
                color='info',
            )),
            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H5(id='label_info3', children=['Total Confirmed'])),
                dbc.CardBody(
                    [
                        html.Div(id= 'New_Cases'),
                        html.H2( id = 'CumConfirmed' ),
                        html.Br()
                    ]
                )

            ],
                color='warning',
            )),

            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H5(id='label_info4', children=['Total Deaths'])),
                dbc.CardBody(
                    [
                        html.Div( id= 'New_Deaths'),
                        html.H2( id='CumDeaths'),
                        html.Div(id='Death_Rate'),
                    ]
                )
            ],
                color='danger',
            )),

            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H5(id='label_info6', children=['Active'])),
                dbc.CardBody(children =[
                    html.Br(),
                    html.H2(id='Active'),
                    html.Br()]
                )
            ],
                color='danger',
            )),


            dbc.Col(dbc.Card([
                dbc.CardHeader(html.H5(id='label_info5', children=['Total Recovered'])),
                dbc.CardBody([
                        html.Div( id= 'New_Recovered'),
                        html.H2( id='CumRecovered'),
                        html.Div(id='Recover_Rate'),
                    ])
            ],
                color='success',
            ))

        ]),
        html.Br(),
        # map
        dbc.Row(  # MIDDLE - MAP & NEWS FEED CONTENT
        [
        # MAPS COL
        dbc.Col(
            # big map
            html.Div(dbc.Spinner(color="primary", type="grow", children=[dcc.Graph(id='graph_map_country')])),
            className="middle-col-map-content",
            width=8,
        ),
        # NEWS FEED COL
        dbc.Col(
            feed_tabs,
            className="left-col-twitter-feed-content",
            width=4,
        ),
    ],
    no_gutters=True,
    className="middle-map-news-content mt-3",
),


        # chart
        html.Div(id='container', children=[], style={'backgroundColor': 'dark'}),
        dbc.Spinner(color="dark", type="grow", children=[html.Button('Add Customized Chart', id='add-chart', n_clicks=1,style={'backgroundColor': '#010914', 'color':'#F4F4F4','border':'1.5px black solid','text-align':'right'})]),
        html.Br(),

        #
        # # WORD CLOUD
        # dcc.Graph(id='graph_wordcloud',figure=A)

    ])
])


################ Navbar  ####################

navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    # dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(
                        dbc.NavbarBrand("COVID-19 Tracker", className="ml-2", style={"font-size": 25})),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://plot.ly",
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
    ],
    color="dark",
    dark=True,
    style={'width': '100%'}
)

# app layout
app.layout = dbc.Container([
    dbc.Row(navbar)
    ,dbc.Tab(tab1, id="label_tab1",label="Dashboard" )
],
    fluid=True
)


################################################################################
#
#    Callbacks
#
################################################################################

@app.callback([
    Output('label_select_criteria', 'children'),
    # Output('label_info1', 'children'),
    Output('label_info2', 'children'),
    Output('label_info3', 'children'),
    Output('label_info4', 'children'),
    Output('label_info5', 'children'),
    Output('label_info6', 'children'),

],
    [Input('label_select_country', 'children')])

def set_lables(label_select_country):

    label_select_country = 'Select the country'
    label_select_criteria = 'Select the criteria'
    # label_info1 = "Country/Region"
    label_info2 = "Date"
    label_info3 = "Total Confirmed"
    label_info4 = "Total Deaths"
    label_info5 = "Total Recovered"
    label_info6 = "Active"
    # label_info7 = "New deaths"
    # label_info8 = "New tests"

    return label_select_criteria, label_info2, label_info3, label_info4, label_info5,label_info6


@app.callback([Output('Date', 'children'),
               Output('CumConfirmed', 'children'),
               Output('CumDeaths', 'children'),
               Output('CumRecovered', 'children'),
               Output('Active', 'children'),
               Output('New_Cases', 'children'),
               Output('New_Deaths', 'children'),
               Output('New_Recovered', 'children'),
               Output('Death_Rate', 'children'),
               Output('Recover_Rate', 'children'),
               # Output('blank', 'children'),
               ],
              [Input('dropdown_country', 'value')])

def info_cards(location_country):
    data_us = data_latest[data_latest['Country/Region'] == location_country]
    Date = data_us['Date'].dt.date
    CumConfirmed = data_us['CumConfirmed']
    CumDeaths = data_us['CumDeaths']
    CumRecovered = data_us['CumRecovered']
    Active = data_us['Active']
    New_Cases = str(list(data_us['New Cases'])[0])
    New_Deaths = str(list(data_us['New Deaths'])[0])
    New_Recovered = str(list(data_us['New Recovered'])[0])
    Death_Rate = str(list(data_us['Death_Rate'])[0])
    Recover_Rate = str(list(data_us['Recover_Rate'])[0])
    # blank = str(' ')
    return Date, CumConfirmed, CumDeaths, CumRecovered, Active, ('+'+ New_Cases + ' '+'New'), ('+'+ New_Deaths+ ' '+'New'), ('+'+ New_Recovered+' '+'New'),('Death Rate:'+ Death_Rate+'%'),('Recover Rate:' + Recover_Rate+'%')



@app.callback(Output('graph_map_country', 'figure'),
              [Input('dropdown_country', 'value'),
               Input('dropdown_criteria', 'value'),
               ])

def map_graph(country, criteria):
    # fig_map = px.choropleth(data_frame=data_latest,
    #                     locations="iso_alpha",
    #                     color=criteria,
    #                     color_continuous_scale=px.colors.sequential.YlOrRd,
    #                     hover_name="Country/Region",
    #                     animation_frame

    country_dict = {}
    for i in range(data_latest.shape[0]):
        country_dict[list(data_latest['Country/Region'])[i]] = {"latitude": list(data_latest['Lat'])[i],
                                                                "longitude": list(data_latest['Long'])[i], "zoom": 3}

    if country == "US":
        lat, lon, zoom = 39.8097343, -98.5556199, 2
    else:
        lat, lon, zoom = (
        country_dict[country]['latitude'],
        country_dict[country]['longitude'],
        country_dict[country]['zoom'],
        )


    fig_map = px.scatter_mapbox(data_latest, lat="Lat", lon="Long", color=criteria, size=criteria,
                                size_max=50, mapbox_style="carto-darkmatter",
                                zoom=1.6, height=500, template=graph_template,
                                color_continuous_scale=px.colors.sequential.YlOrRd , animation_frame = 'Date',animation_group = 'Country/Region', hover_name='Country/Region', hover_data=
                                ['CumConfirmed', 'CumDeaths', 'CumRecovered','Active', 'New Cases', 'New Deaths', 'New Recovered',
                                 'Death_Rate','Recover_Rate']
                                )

    # fig_map.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 4000
    # fig_map.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 0.000000004

    text_title = f'Map of {criteria} of COVID-19 by country - {country}'
    text_xaxis = 'Date'
    text_yaxis = criteria
    text_legend = 'Country/Region'
    fig_map.update_layout(
        coloraxis_showscale=True,
        mapbox_style="carto-darkmatter",
        mapbox=dict(center=dict(lat=lat, lon=lon), zoom=zoom),
        title=text_title
        # , xaxis_title=text_xaxis,
        #                   yaxis_title=text_yaxis, legend_title=text_legend
    )
    # fig_map.update_layout(title=text_title, xaxis_title=text_xaxis,
    #                       yaxis_title=text_yaxis, legend_title=text_legend)

    return fig_map


# @app.callback([Output('graph_line_country', 'figure')
#                   # ,Output('graph_area_country', 'figure')
#                ],
#               [Input('dropdown_country', 'value'),
#                Input('dropdown_criteria', 'value'),
#                ])
# def line_graphs(country, criteria):
#     # line graph - continent
#     data = full_grouped
#     full_grouped['Date'] = full_grouped['Date'].apply(str)
#     line_fig = px.line(data, x='Date', y=criteria, color='Country/Region', template=graph_template)
#
#
#     text_title = f'Time line of {criteria} of Covid 19 by Country - {country}'
#     text_xaxis = 'Date'
#     text_yaxis = criteria
#     text_legend = 'Country/Region'
#     line_fig.update_traces(textfont_size=30)
#
#     line_fig.update_layout(title=text_title[0], xaxis_title=text_xaxis[0],
#                            yaxis_title=text_yaxis[0], legend_title=text_legend[0])
#

# Add and display customized charts
@app.callback(
    Output('container', 'children'),
    [Input('add-chart', 'n_clicks')],
    [State('container', 'children')]
)
def display_graphs(n_clicks, div_children):
    new_child = html.Div(
        style={'width': '100%', 'backgroundColor':"#19202A" },
        children=[
            dcc.Graph(
                id={
                    'type': 'dynamic-graph',
                    'index': n_clicks,
                },
                figure={'layout': go.Layout(
            paper_bgcolor='#010914',
            plot_bgcolor='#010914')}
            ),
            dcc.RadioItems(
                id={
                    'type': 'dynamic-choice',
                    'index': n_clicks
                },
                options=[{'label': 'Bar Chart', 'value': 'bar'},
                         {'label': 'Line Chart', 'value': 'line'}],
                value='line',
            ),
            dcc.Dropdown(
                id={
                    'type': 'dynamic-dpn-s',
                    'index': n_clicks
                },
                options=[{'label': s, 'value': s} for s in np.sort(data_latest['Country/Region'].unique())],
                multi=True,
                value=data_latest.nlargest(10, 'CumConfirmed', keep='last')['Country/Region']
,style={'backgroundColor':'#F4F4F4', 'color':'black'}
            ),
            dcc.Dropdown(
                id={
                    'type': 'dynamic-dpn-num',
                    'index': n_clicks
                },
                options=[{'label': n, 'value': n} for n in
                         ['CumConfirmed',
 'CumDeaths',
 'CumRecovered',
 'Active',
 'New Cases',
 'New Deaths',
 'New Recovered',
 'Death_Rate',
 'Recover_Rate']],
                value='CumConfirmed',style={'backgroundColor':'#F4F4F4', 'color':'black'},
                clearable=False
            )
        ]
    )
    div_children.append(new_child)
    return div_children


# Return different options for bar chart and line chart
@app.callback(
    [Output({'type': 'dynamic-dpn-num', 'index': MATCH}, 'options'),
     Output({'type': 'dynamic-dpn-num', 'index': MATCH}, 'value')],
    [Input({'type': 'dynamic-choice', 'index': MATCH}, 'value')]
)
def choose_graph(chart_choice):
    if chart_choice == 'bar':
        option = [{'label': n, 'value': n} for n in
                  ['CumConfirmed',
 'CumDeaths',
 'CumRecovered',
 'Active',
 'New Cases',
 'New Deaths',
 'New Recovered',
 'Death_Rate',
 'Recover_Rate']]
        return option, 'CumConfirmed'
    elif chart_choice == 'line':
        option = [{'label': n, 'value': n} for n in ['CumConfirmed',
 'CumDeaths',
 'CumRecovered',
 'Active',
 'New Cases',
 'New Deaths',
 'New Recovered',
 'Death_Rate',
 'Recover_Rate']]
        return option, 'CumConfirmed'


# Update each added chart
@app.callback(
    Output({'type': 'dynamic-graph', 'index': MATCH}, 'figure'),
    [Input(component_id={'type': 'dynamic-dpn-s', 'index': MATCH}, component_property='value'),
     Input(component_id={'type': 'dynamic-dpn-num', 'index': MATCH}, component_property='value'),
     Input({'type': 'dynamic-choice', 'index': MATCH}, 'value')]
)
def update_graph(s_value, num_value, chart_choice):
    if chart_choice == 'bar':
        dfbar = data_latest[data_latest['Country/Region'].isin(s_value)]
        fig = px.bar(dfbar, x='Country/Region', y=num_value,color=num_value,color_continuous_scale=px.colors.sequential.YlOrRd,template = graph_template,
                     title="{} Across Countries".format(num_value)).update_layout(
            xaxis={'categoryorder': 'total descending'})
        fig.update_layout(
            font_family="Arial",
            font_color="white",
            title_font_family="Arial",
            title_font_color="white",
            legend_title_font_color="white",
            title={'y': 0.9,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'top'},
            font=dict(
                family="Calibri",
                size=15,
                color="white")
            )

        return fig
    elif chart_choice == 'line':
        if len(s_value) == 0:
            return {}
        else:
            dfline = full_grouped[full_grouped['Country/Region'].isin(s_value)]
            fig = px.line(dfline, x='Date', y=num_value, color='Country/Region',template = graph_template, title="{} Across Countries".format(num_value),
                          range_x=['2020-01-23', max(full_grouped['Date'])])
            fig.update_layout(
                font_family="Arial",
                font_color="white",
                title_font_family="Arial",
                title_font_color="white",
                legend_title_font_color="white",
                title={'y': 0.9,
                       'x': 0.5,
                       'xanchor': 'center',
                       'yanchor': 'top'},
                font=dict(
                    family="Calibri",
                    size=15,
                    color="white"
                )
            )
            return fig

        return fig

########################################################################
#
#    News callbacks
#
############################################################################


def news_feed( country) :

    newsapi = NewsApiClient(api_key='6c7bb4a85ec049b3b9459379a35a36fa')
    q = 'corona'
    json_data = newsapi.get_top_headlines(q=q, language='en', country='us')
    # country_name_code_dict.get(str(country)
    data = pd.DataFrame(json_data['articles'])

    source = data['source'][0].get('name')
    source = []
    for i in range(data.shape[0]):
        news = data['source'][i].get('name')
        source.append(news)
    news_data = pd.DataFrame(data[["title", "url"]])
    news_data['source'] = source


    max_rows = 50
    list_group = dbc.ListGroup(
        [
            dbc.ListGroupItem(
                [
                    html.Div(
                        [
                            html.H6(
                                f"{news_data.iloc[i]['title'].split(' - ')[0]}.",
                                className="news-txt-headline",
                            ),
                            html.P(
                                f"{news_data.iloc[i]['title'].split(' - ')[1]}"
                                f"  {news_data.iloc[i]['source']}",
                                className="news-txt-by-dt",
                            ),
                        ],
                        className="news-item-container",
                    )
                ],
                className="news-item",
                href=news_data.iloc[i]["url"],
                target="_blank",
            )
            for i in range(min(len(news_data),max_rows))
        ],
        flush=True,
    )


    return list_group


    # del news_data
@app.callback(
        [
         Output("feed_content", "children")],
        [
            Input('label_select_country', 'children'),
        ],
    )  # pylint: disable=W0612
def feed_tab_content(a):
    return [news_feed(a)]



if __name__ == '__main__':
    app.run_server(debug=False)
