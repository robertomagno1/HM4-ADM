import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


class KMeansVisualizer:
    def __init__(self, data, centroids_history, labels_history):
        """
        Visualizza i risultati dell'algoritmo K-Means/K-Means++ con storia.

        Args:
        - data: Dataset originale (array NumPy, 2D)
        - centroids_history: Lista delle posizioni dei centroidi per ogni iterazione
        - labels_history: Lista delle assegnazioni di cluster per ogni iterazione
        """
        self.data = data
        self.centroids_history = centroids_history
        self.labels_history = labels_history
        self.iteration_data = []
        
        # Prepara i dati per la visualizzazione
        self._prepare_visualization_data()
    
    def _prepare_visualization_data(self):
        """
        Prepara i dati per ogni iterazione per la visualizzazione interattiva.
        """
        for iteration, (centroids, labels) in enumerate(zip(self.centroids_history, self.labels_history)):
            for i, point in enumerate(self.data):
                self.iteration_data.append({
                    'Iteration': iteration + 1,
                    'Feature1': point[0],
                    'Feature2': point[1],
                    'Cluster': labels[i],
                    'Centroid_X': centroids[labels[i]][0],
                    'Centroid_Y': centroids[labels[i]][1]
                })
    
    def create_visualization(self):
        """
        Crea la visualizzazione interattiva dell'evoluzione dei cluster.
        """
        # Converti i dati in DataFrame
        df_iterations = pd.DataFrame(self.iteration_data)
        
        # Colori per i cluster
        color_map = px.colors.qualitative.Plotly
        
        # Prepara i frame per l'animazione
        frames = []
        for iteration in df_iterations['Iteration'].unique():
            df_frame = df_iterations[df_iterations['Iteration'] == iteration]
            
            # Scatter dei punti
            scatter = go.Scatter(
                x=df_frame['Feature1'], 
                y=df_frame['Feature2'],
                mode='markers',
                marker=dict(
                    size=10, 
                    color=[color_map[c % len(color_map)] for c in df_frame['Cluster']],
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name=f'Punti - Iterazione {iteration}'
            )
            
            # Scatter dei centroidi
            centroids = go.Scatter(
                x=df_frame['Centroid_X'], 
                y=df_frame['Centroid_Y'],
                mode='markers',
                marker=dict(
                    symbol='x', 
                    size=15, 
                    color='red', 
                    line=dict(width=2)
                ),
                name=f'Centroidi - Iterazione {iteration}'
            )
            
            # Aggiungi frame
            frame = go.Frame(
                data=[scatter, centroids],
                name=f'frame{iteration}'
            )
            frames.append(frame)
        
        # Crea figura iniziale
        fig = make_subplots(
            rows=1, cols=1, 
            subplot_titles=('Evoluzione del clustering K-Means++')
        )
        first_frame = frames[0]
        fig.add_trace(first_frame.data[0])
        fig.add_trace(first_frame.data[1])
        
        # Configura animazione
        fig.frames = frames
        fig.update_layout(
            title='Evoluzione del clustering K-Means++',
            xaxis_title='Caratteristica 1',
            yaxis_title='Caratteristica 2',
            height=800,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Avvia',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 1000, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300}
                        }]
                    },
                    {
                        'label': 'Pausa',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': 'Ripristina',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 1000, 'redraw': True},
                            'mode': 'immediate',
                            'fromcurrent': False,
                            'transition': {'duration': 300}
                        }]
                    }
                ]
            }]
        )
        
        return fig



