import pandas as pd
import streamlit as st

import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch
from highlight_text import  fig_text
from soccerplots.utils import add_image
import datetime

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

st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def ETL_GPS():
    def toCSV(url):
        df = pd.read_csv(url, delimiter=';', skiprows=8)

        return df

    data = []
    for file in tqdm.tqdm(os.listdir(r'./Data')):

        gpsData = toCSV('./Data/' + file)

        gpsData['gameTime'] = gpsData['Timestamp'].apply(lambda x: x.split(' ')[1])
        gpsData['Day'] = gpsData['Timestamp'].apply(lambda x: x.split(' ')[0])

        gpsData['Velocity'] = gpsData['Velocity'].apply(lambda x: x.replace(',', '.')).astype(float)

        gpsData['Longitude'] = gpsData['Longitude'].apply(lambda x: x.replace(',', '.')).astype(float)
        gpsData['Latitude'] = gpsData['Latitude'].apply(lambda x: x.replace(',', '.')).astype(float)
        gpsData['Player Load'] = gpsData['Player Load'].apply(lambda x: x.replace(',', '.')).astype(float)

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

game_direction = st.sidebar.text_input('Attacking Direction', 'Left')

playerLoad = playerGPS.loc[playerGPS.Player == selected_player]['Player Load'].tail().reset_index(drop=True)

maxVelocity = playerGPS.loc[(playerGPS.Player == selected_player) & (playerGPS['Velocity'] >= 25)]['Velocity'].max()
if select_half == 'First':
        maxVelocityHalf = playerGPS.loc[(playerGPS.Player == selected_player) & (playerGPS['Velocity'] >= 25) &
                                        (playerGPS['gameTime'] > start_game) & (playerGPS['gameTime'] <= half_end1st)]['Velocity'].max()
        
elif select_half == 'Second':
        maxVelocityHalf = playerGPS.loc[(playerGPS.Player == selected_player) & (playerGPS['Velocity'] >= 25) &
                                        (playerGPS['gameTime'] > half_start2nd) & (playerGPS['gameTime'] <= end_game)]['Velocity'].max()

def identify_sprints(df, player_name, day):
    player_df = df[(df['Player'] == player_name) & (df['Day'] == day) &
                   (df['gameTime'] > start_game) & (df['gameTime'] <= end_game)].reset_index(drop=True)

    is_sprint = player_df['Velocity'] >= 25
    sprint_sequence = (is_sprint.diff() != 0).cumsum() - 1
    sprint_sequence[~is_sprint] = 0
    player_df['isSprint'] = is_sprint
    player_df['Sequence'] = sprint_sequence
    player_df['startX'] = np.nan
    player_df['startY'] = np.nan
    player_df['xEnd'] = np.nan
    player_df['yEnd'] = np.nan

    for sprint_num in player_df['Sequence'].unique():
        sprint_mask = player_df['Sequence'] == sprint_num
        if sprint_mask.any():
            start_row = player_df.loc[sprint_mask].iloc[0]
            end_row = player_df.loc[sprint_mask].iloc[-1]
            player_df.loc[sprint_mask, 'startX'] = start_row['x']
            player_df.loc[sprint_mask, 'startY'] = start_row['y']
            player_df.loc[sprint_mask, 'xEnd'] = end_row['x']
            player_df.loc[sprint_mask, 'yEnd'] = end_row['y']

    return player_df

sprintCount = identify_sprints(playerGPS, selected_player, selected_Day)

sprints = len(sprintCount.Sequence.unique()[1:])
if select_half == 'First':
        sprintsHalf = len(sprintCount.loc[(sprintCount['Velocity'] >= 25) &
                                          (sprintCount['gameTime'] > start_game) &
                                          (sprintCount['gameTime'] <= half_end1st)]['Sequence'].unique())
        
elif select_half == 'Second':
        sprintsHalf = len(sprintCount.loc[(sprintCount.Player == selected_player) &
                                        (sprintCount['Velocity'] >= 25) &
                                        (sprintCount['gameTime'] > half_start2nd) &
                                        (sprintCount['gameTime'] <= end_game)]['Sequence'].unique())

st.text('The value of the box is the half value compared with the overall game value.')

col1, col2, col3 = st.columns(3)
col1.metric("Player Load", playerLoad[4], delta_color="off")
col2.metric("Max Velocity", maxVelocityHalf, maxVelocity, delta_color="off")
col3.metric("Nº Sprints", sprintsHalf, sprints, delta_color="off")

col1, col2 = st.columns(2)
lenght = col1.text_input('Pitch length', '106')

width = col2.text_input('Pitch width', '74')

# Funtion to generate GPS Player HeatMap
st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def catapultHeatMap(df, playerName, matchDay, halfGame, startGame, halfBreak1st, halfBreak2nd, endGame, direction):
    
        # Load GPS DATA (CSV FILE)
        #data = pd.read_csv(filePath, delimiter=';')
        
        data = df.loc[(df.Player == playerName) & (df.Day == matchDay)].reset_index(drop=True)
        
        # CREATE PITCH USING MPLSOCCER LIBRARY
        pitch = Pitch(pitch_type='metricasports', line_zorder=2,
                        pitch_length=int(lenght), pitch_width=int(width),
                        pitch_color='#E8E8E8', line_color='#181818',
                        corner_arcs=True, goal_type='box')

        fig, ax = pitch.draw(figsize=(15, 10))
        fig.set_facecolor('#E8E8E8')

        # GRADIENT COLOR FOR THE HEATMAP
        pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                                ['#E8E8E8', '#FF0000'], N=10)


        # FUNCTIONS FROM MPLSOCCER TO CREATE A HEATMAP

        if halfGame == 'First':
                heatMap = data.loc[(data['gameTime'] >= startGame) & (data['gameTime'] <= halfBreak1st)].reset_index(drop=True)
        elif halfGame == 'Second':
                heatMap = data.loc[(data['gameTime'] > halfBreak2nd) & (data['gameTime'] <= endGame)].reset_index(drop=True)

        bs = pitch.bin_statistic_positional(heatMap['y'], heatMap['x'],  statistic='count', positional='full', normalize=True)
        
        pitch.heatmap_positional(bs, edgecolors='#e8e8e8', ax=ax, cmap=pearl_earring_cmap, alpha=0.6)
        
        path_eff = [path_effects.Stroke(linewidth=3, foreground='black'),
                        path_effects.Normal()]

        pitch.label_heatmap(bs, color='#E8E8E8', fontsize=25,
                                ax=ax, ha='center', va='center',
                                str_format='{:.0%}', path_effects=path_eff)
        #Params for the text inside the <> this is a function to highlight text
        highlight_textprops =\
                [{"color": '#FF0000',"fontweight": 'bold'}]

        # TEXT: NAME OF THE PLAYER AND THE GAME
        fig_text(s = playerName + ' <HeatMap>', highlight_textprops=highlight_textprops, x = 0.5, y = 1.105, color='#181818', ha='center', fontsize=50);

        fig_text(s = halfGame + ' Half', x = 0.5, y = 1.03, color='#181818', ha='center', fontsize=20);

        fig_text(s = matchDay, x = 0.5, y = 1, color='#181818', ha='center', fontsize=14);

        # LOGO OF THE CLUB
        fig = add_image(image='./Images/Clubs/Brasileirao/Ceara.png', fig=fig, left=0.1, bottom=0.985, width=0.15, height=0.12)

        if direction == 'Left':
                fig_text(s = 'Attacking Direction',
                        x = 0.5, y = 0.07,
                        color='#181818', fontweight='bold',
                        ha='center', va='center',
                        fontsize=14)

                # ARROW DIRECTION OF PLAY
                ax.annotate('', xy=(0.3, -0.07), xycoords='axes fraction', xytext=(0.7, -0.07), 
                        arrowprops=dict(arrowstyle="<-", color='#181818', lw=2))

        elif direction == 'Right':
                fig_text(s = 'Attacking Direction',
                        x = 0.5, y = 0.07,
                        color='#181818', fontweight='bold',
                        ha='center', va='center',
                        fontsize=14)

                # ARROW DIRECTION OF PLAY
                ax.annotate('', xy=(0.3, -0.07), xycoords='axes fraction', xytext=(0.7, -0.07), 
                        arrowprops=dict(arrowstyle="->", color='#181818', lw=2))

        return plt.show()
figHeatMap = catapultHeatMap(playerGPS, selected_player, selected_Day, select_half, start_game, half_end1st, half_start2nd, end_game, game_direction)
st.title('HeatMap')
st.pyplot(figHeatMap)

st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def plotSprints(df, playerName, matchDay, halfGame, startGame, halfBreak1st, halfBreak2nd, endGame, direction):
        
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
                sprints = data.loc[(data['Velocity'] >= 25) & (data['gameTime'] >= startGame) & (data['gameTime'] <= halfBreak1st)].reset_index(drop=True)
        elif halfGame == 'Second':
                sprints = data.loc[(data['Velocity'] >= 25) & (data['gameTime'] > halfBreak2nd) & (data['gameTime'] <= endGame)].reset_index(drop=True)
                
        #Criação das setas que simbolizam os passes realizados bem sucedidos
        pitch.arrows(sprints['startY'], sprints['startX'],
                     sprints['yEnd'], sprints['xEnd'], width=2,
                     headwidth=10, headlength=10, color='#181818', ax=ax, label='Sprints')

        #Params for the text inside the <> this is a function to highlight text
        highlight_textprops =\
                [{"color": '#FF0000',"fontweight": 'bold'}]

        # TEXT: NAME OF THE PLAYER AND THE GAME
        fig_text(s = playerName + ' <Sprints>', highlight_textprops=highlight_textprops, x = 0.5, y = 1.105, color='#181818', ha='center', fontsize=50);
        
        fig_text(s = halfGame + ' Half', x = 0.5, y = 1.03, color='#181818', ha='center', fontsize=20);

        fig_text(s = matchDay, x = 0.5, y = 1, color='#181818', ha='center', fontsize=14);

        # LOGO OF THE CLUB
        fig = add_image(image='./Images/Clubs/Brasileirao/Ceara.png', fig=fig, left=0.1, bottom=0.985, width=0.15, height=0.12)
        
        if direction == 'Left':
                fig_text(s = 'Attacking Direction',
                        x = 0.5, y = 0.07,
                        color='#181818', fontweight='bold',
                        ha='center', va='center',
                        fontsize=14)

                # ARROW DIRECTION OF PLAY
                ax.annotate('', xy=(0.3, -0.07), xycoords='axes fraction', xytext=(0.7, -0.07), 
                        arrowprops=dict(arrowstyle="<-", color='#181818', lw=2))

        elif direction == 'Right':
                fig_text(s = 'Attacking Direction',
                        x = 0.5, y = 0.07,
                        color='#181818', fontweight='bold',
                        ha='center', va='center',
                        fontsize=14)

                # ARROW DIRECTION OF PLAY
                ax.annotate('', xy=(0.3, -0.07), xycoords='axes fraction', xytext=(0.7, -0.07), 
                        arrowprops=dict(arrowstyle="->", color='#181818', lw=2))
        
        return plt.show()
figSprints = plotSprints(sprintCount, selected_player, selected_Day, select_half, start_game, half_end1st, half_start2nd, end_game, game_direction)
st.title('Sprints')
st.pyplot(figSprints)

st.cache_data(ttl=datetime.timedelta(hours=1), max_entries=1000)
def catapultMeanPos(df, playerName, matchDay, halfGame, startGame, halfBreak1st, halfBreak2nd, endGame, direction):
    
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
                player = data.loc[(data['gameTime'] >= startGame) & (data['gameTime'] <= halfBreak1st)].reset_index(drop=True)
        elif halfGame == 'Second':
                player = data.loc[(data['gameTime'] > halfBreak2nd) & (data['gameTime'] <= endGame)].reset_index(drop=True)

        pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",
                                                        ['#E8E8E8','#FF0000'], N=10)

        bs = pitch.bin_statistic(player['y'], player['x'], bins=(10, 8))

        convex = player[(np.abs(stats.zscore(player[['y','x']])) < 0.8).all(axis=1)]

        pitch.heatmap(bs, edgecolors='#E8E8E8', ax=ax, cmap=pearl_earring_cmap)

        hull = pitch.convexhull(convex['y'], convex['x'])

        pitch.polygon(hull, ax=ax, edgecolor='#181818', facecolor='#181818', alpha=0.7, linestyle='--', linewidth=2.5)

        pitch.scatter(x=convex['y'].mean(), y=convex['x'].mean(), ax=ax, c='#E8E8E8', edgecolor='#FF0000', lw=1.5, s=700, zorder=4)

        #Params for the text inside the <> this is a function to highlight text
        highlight_textprops =\
                [{"color": '#FF0000',"fontweight": 'bold'}]

        # TEXT: NAME OF THE PLAYER AND THE GAME
        fig_text(s = playerName + ' <Average Position>', highlight_textprops=highlight_textprops, x = 0.52, y = 1.105, color='#181818', ha='center', fontsize=50);

        fig_text(s = 'All game', x = 0.5, y = 1.03, color='#181818', ha='center', fontsize=20);

        fig_text(s = matchDay, x = 0.5, y = 1, color='#181818', ha='center', fontsize=14);

        # LOGO OF THE CLUB
        fig = add_image(image='./Images/Clubs/Brasileirao/Ceara.png', fig=fig, left=0.1, bottom=0.985, width=0.15, height=0.12)

        if direction == 'Left':
                fig_text(s = 'Attacking Direction',
                        x = 0.5, y = 0.07,
                        color='#181818', fontweight='bold',
                        ha='center', va='center',
                        fontsize=14)

                # ARROW DIRECTION OF PLAY
                ax.annotate('', xy=(0.3, -0.07), xycoords='axes fraction', xytext=(0.7, -0.07), 
                        arrowprops=dict(arrowstyle="<-", color='#181818', lw=2))

        elif direction == 'Right':
                fig_text(s = 'Attacking Direction',
                        x = 0.5, y = 0.07,
                        color='#181818', fontweight='bold',
                        ha='center', va='center',
                        fontsize=14)

                # ARROW DIRECTION OF PLAY
                ax.annotate('', xy=(0.3, -0.07), xycoords='axes fraction', xytext=(0.7, -0.07), 
                        arrowprops=dict(arrowstyle="->", color='#181818', lw=2))
                
        return plt.show()

figMeanPos = catapultMeanPos(playerGPS, selected_player, selected_Day, select_half, start_game, half_end1st, half_start2nd, end_game, game_direction)
st.title('Average position')
st.pyplot(figMeanPos)

