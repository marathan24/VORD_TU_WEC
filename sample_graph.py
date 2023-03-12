import numpy as np
from sklearn.manifold import MDS
import plotly.graph_objs as go

cosine_similarities = np.array([[1.0, 0.8, 0.4, 0.2, 0.1],
                               [0.8, 1.0, 0.6, 0.3, 0.2],
                               [0.4, 0.6, 1.0, 0.5, 0.4],
                               [0.2, 0.3, 0.5, 1.0, 0.8],
                               [0.1, 0.2, 0.4, 0.8, 1.0]])


mds = MDS(n_components=3, dissimilarity='precomputed')
coords = mds.fit_transform(1 - cosine_similarities)

fig = go.Figure()
for i, coord in enumerate(coords):
    fig.add_trace(go.Scatter3d(
        x=[coord[0]],
        y=[coord[1]],
        z=[coord[2]],
        mode='markers',
        marker=dict(
            size=10,
            color='blue',
            opacity=0.8
        ),
        text=f'Word {i+1}',
        name=f'Word {i+1}'
    ))


fig.update_layout(
    title='Words in 3D space',
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
    ),
    hovermode='closest'
)


frames = []
for i in range(len(coords)):
    frame = go.Frame(
        data=[go.Scatter3d(
            x=[coord[0]],
            y=[coord[1]],
            z=[coord[2]],
            mode='markers',
            marker=dict(
                size=10,
                color='blue',
                opacity=0.8
            ),
            text=f'Word {i+1}',
            name=f'Word {i+1}'
        ) for coord in coords[:i+1]],
        name=f'Frame {i+1}'
    )
    frames.append(frame)

fig.frames = frames
fig.update_layout(updatemenus=[dict(
    type='buttons',
    showactive=False,
    buttons=[dict(
        label='Play',
        method='animate',
        args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]
    ), dict(
        label='Pause',
        method='animate',
        args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate', transition=dict(duration=0))]
    )]
)])


fig.show()
