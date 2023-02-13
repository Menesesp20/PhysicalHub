import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch
from highlight_text import  fig_text
from soccerplots.utils import add_image

from matplotlib import font_manager

import tqdm as tqdm
import os

font_path = './Fonts/Gagalin-Regular.otf'
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

# Courier New
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Ceará Players GPS Hub')

st.markdown("""
This app performs visualization from GPS data!
* **Data source:** Catapult.
""")

st.sidebar.header('Physical Hub')

st.title('Upload Data')
def save_uploadedfile(uploadedfile):
     with open(os.path.join("Data", uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to Data".format(uploadedfile.name))
 
gps_file = st.file_uploader("Upload a file", type=['csv','xlsx'])
if gps_file is not None:
   file_details = {"FileName":gps_file.name, "FileType":gps_file.type}
   df  = pd.read_csv(gps_file, delimiter=';', skiprows=8)
   st.dataframe(df)
   save_uploadedfile(gps_file)

st.cache_data()
def ETL_GPS():
    def toCSV(url):
        df = pd.read_csv(url, delimiter=';', skiprows=8)

        return df

    playerNames = []
    data = []
    for file in tqdm.tqdm(os.listdir(r'./Data')):

        gameDay = file.split(' ')[0] + file.split(' ')[1]
        #playerNames.append(file.split(' ')[6] + file.split(' ')[7])

        gpsData = toCSV('./Data/' + file)

        gpsData['gameTime'] = gpsData['Timestamp'].apply(lambda x: x.split(' ')[1])
        gpsData['Day'] = gpsData['Timestamp'].apply(lambda x: x.split(' ')[0])

        gpsData['Velocity'] = gpsData['Velocity'].apply(lambda x: x.replace(',', '.')).astype(float)

        gpsData['Longitude'] = gpsData['Longitude'].apply(lambda x: x.replace(',', '.')).astype(float)
        gpsData['Latitude'] = gpsData['Latitude'].apply(lambda x: x.replace(',', '.')).astype(float)

        # VARIABLES TO USE THE MIN AND MAX VALUES AND CONVERT THE DATA WICH IS NEGATIVE TO A RANGE BETWEEN 0 AND 1
        lower_bound = gpsData['Longitude'].min()
        upper_bound = gpsData['Longitude'].max()

        gpsData['x'] = (gpsData['Longitude'] - lower_bound) / (upper_bound - lower_bound)

        lower_bound = gpsData['Latitude'].min()
        upper_bound = gpsData['Latitude'].max()
        
        gpsData['y'] = (gpsData['Latitude'] - lower_bound) / (upper_bound - lower_bound)
        
        gpsData['Player'] = file.split(' ')[6]  + ' ' + file.split(' ')[7]

        concatData = pd.concat([gpsData], axis=0, ignore_index=True)

        data.append(concatData)
        
    df = pd.concat(data)
    
    return df
playerGPS = ETL_GPS()

# Sidebar - Player selection
sorted_unique_Player = sorted(playerGPS.Player.unique())
selected_player = st.sidebar.selectbox('Players:', sorted_unique_Player)

# Sidebar - GameDay selection
sorted_unique_Day = sorted(playerGPS.Day.unique())
selected_Day = st.sidebar.selectbox('MatchDay:', sorted_unique_Day)

half_option = ['First', 'Second']
select_half = st.sidebar.selectbox('Period', half_option)

start_game = st.sidebar.text_input('Start Hours', '17:30:00')

half_end1st = st.sidebar.text_input('First Half Hours', '18:18:00')

half_start2nd = st.sidebar.text_input('Second Half Hours', '18:35:00')

end_game = st.sidebar.text_input('End Hours', '19:22:00')

# Funtion to generate GPS Player HeatMap
def catapultHeatMap(df, playerName, matchDay, halfGame, startGame, halfBreak1st, halfBreak2nd, endGame):
    
    # Load GPS DATA (CSV FILE)
    #data = pd.read_csv(filePath, delimiter=';')
    
    data = df.loc[(df.Player == playerName) & (df.Day == matchDay)].reset_index(drop=True)
    
    # CREATE PITCH USING MPLSOCCER LIBRARY
    pitch = Pitch(pitch_type='metricasports', line_zorder=2,
                pitch_length=106, pitch_width=74,
                pitch_color='#E8E8E8', line_color='#181818',
                corner_arcs=True, goal_type='box')

    fig, ax = pitch.draw(figsize=(15, 10))
    fig.set_facecolor('#E8E8E8')

    # GRADIENT COLOR FOR THE HEATMAP
    pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                            ['#E8E8E8', '#FF0000'], N=10)


    # FUNCTIONS FROM MPLSOCCER TO CREATE A HEATMAP

    if halfGame == 'First':
            heatMap = data.loc[(data['Velocity'] >= 6.94) & (data['gameTime'] >= startGame) & (data['gameTime'] <= halfBreak1st)].reset_index(drop=True)
    elif halfGame == 'Second':
            heatMap = data.loc[(data['Velocity'] >= 6.94) & (data['gameTime'] > halfBreak2nd) & (data['gameTime'] == endGame)].reset_index(drop=True)

    bs = pitch.bin_statistic(heatMap['x'], heatMap['y'], bins=(10, 8))
    pitch.heatmap(bs, edgecolors='#E8E8E8', ax=ax, cmap=pearl_earring_cmap)

    #Params for the text inside the <> this is a function to highlight text
    highlight_textprops =\
            [{"color": '#FF0000',"fontweight": 'bold'}]

    # TEXT: NAME OF THE PLAYER AND THE GAME
    fig_text(s = playerName + ' <HeatMap>', highlight_textprops=highlight_textprops, x = 0.5, y = 1.105, color='#181818', ha='center', fontsize=50);

    fig_text(s = halfGame + ' Half', x = 0.5, y = 1.03, color='#181818', ha='center', fontsize=20);

    fig_text(s = 'Ceará SC vs Sampaio Corrêa FC | 04/02/2023', x = 0.5, y = 1, color='#181818', ha='center', fontsize=14);

    # LOGO OF THE CLUB
    fig = add_image(image='./Images/Clubs/Brasileirao/Ceara.png', fig=fig, left=0.1, bottom=0.985, width=0.15, height=0.12)
    
    return plt.show()
figHeatMap = catapultHeatMap(playerGPS, selected_player, selected_Day, select_half, start_game, half_end1st, half_start2nd, end_game)

st.title('HeatMap')
st.pyplot(figHeatMap)

def plotSprints(df, playerName, matchDay, halfGame, startGame, halfBreak1st, halfBreak2nd, endGame):
        
        #data = pd.read_csv(filePath, delimiter=';')

        data = df.loc[(df.Player == playerName) & (df.Day == matchDay)].reset_index(drop=True)

        # CREATE PITCH USING MPLSOCCER LIBRARY
        pitch = Pitch(pitch_type='metricasports', line_zorder=2,
                pitch_length=106, pitch_width=74,
                pitch_color='#E8E8E8', line_color='#181818',
                corner_arcs=True, goal_type='box')

        fig, ax = pitch.draw(figsize=(15, 10))
        fig.set_facecolor('#E8E8E8')

        if halfGame == 'First':
                sprints = data.loc[(data['Velocity'] >= 6.94) & (data['gameTime'] >= startGame) & (data['gameTime'] <= halfBreak1st)].reset_index(drop=True)
        elif halfGame == 'Second':
                sprints = data.loc[(data['Velocity'] >= 6.94) & (data['gameTime'] > halfBreak2nd) & (data['gameTime'] == endGame)].reset_index(drop=True)
                
        #Criação das setas que simbolizam os passes realizados bem sucedidos
        pitch.scatter(sprints['x'], sprints['y'], color='#181818', ax=ax)

        #Params for the text inside the <> this is a function to highlight text
        highlight_textprops =\
                [{"color": '#FF0000',"fontweight": 'bold'}]

        # TEXT: NAME OF THE PLAYER AND THE GAME
        fig_text(s = playerName + ' <Sprints>', highlight_textprops=highlight_textprops, x = 0.5, y = 1.105, color='#181818', ha='center', fontsize=50);
        
        fig_text(s = halfGame + ' Half', x = 0.5, y = 1.03, color='#181818', ha='center', fontsize=20);

        fig_text(s = 'Ceará SC vs Sampaio Corrêa FC | 04/02/2023', x = 0.5, y = 1, color='#181818', ha='center', fontsize=14);

        # LOGO OF THE CLUB
        fig = add_image(image='./Images/Clubs/Brasileirao/Ceara.png', fig=fig, left=0.1, bottom=0.985, width=0.15, height=0.12)
        
        return plt.show()
figSprints = plotSprints(playerGPS, selected_player, selected_Day, select_half, start_game, half_end1st, half_start2nd, end_game)

st.title('Sprints')
st.pyplot(figSprints)
playerGPS = ETL_GPS()