from dash.dependencies import Output, Input
from flask import Flask, request
import plotly.graph_objs as go
from datetime import datetime
from collections import deque
from dash import dcc, html
import json
import dash
import os

# Flask server setup
server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# Deque setup for storing data points
MAX_DATA_POINTS = 1000
UPDATE_FREQ_MS = 100  # frequency of graph updates in milliseconds

# Deques for time and accelerometer data
time = deque(maxlen=MAX_DATA_POINTS)
accel_x = deque(maxlen=MAX_DATA_POINTS)
accel_y = deque(maxlen=MAX_DATA_POINTS)
accel_z = deque(maxlen=MAX_DATA_POINTS)

# Dash app layout
app.layout = html.Div(
    [
        dcc.Markdown("# Live Sensor Readings"),
        dcc.Graph(id="live_graph", style={"height": "60vh"}),
        dcc.Interval(id="counter", interval=UPDATE_FREQ_MS),
    ]
)

# Callback to update the graph
@app.callback(Output("live_graph", "figure"), Input("counter", "n_intervals"))
def update_graph(_):
    print("Updating graph...")  # Debug print statement
    # Create the graph with the data
    traces = [
        go.Scatter(x=list(time), y=list(d), mode='lines', name=name)
        for d, name in zip([accel_x, accel_y, accel_z], ["X", "Y", "Z"])
    ]
    layout = go.Layout(
        xaxis=dict(title='Time', type='date'),
        yaxis=dict(title='Acceleration (m/s²)'),
        margin=dict(l=50, r=50, b=50, t=50),
        showlegend=True
    )
    return {'data': traces, 'layout': layout}

# Endpoint to receive data and save it
@app.server.route("/data", methods=["POST"])
def receive_data():
    if request.method == "POST":
        raw_data = request.get_data(as_text=True)
        print(f"Received data: {raw_data}")  # Debug print statement
        
        # Save the raw data to a file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"data/data_{timestamp}.json"
        with open(filename, 'w') as file:
            file.write(raw_data)

        # Parse the JSON data and update deques
        data_json = json.loads(raw_data)
        for d in data_json.get('payload', []):
            if d.get("name") == "accelerometer":
                ts = datetime.fromtimestamp(d["time"] / 1e9)  # Convert nanoseconds to seconds
                time.append(ts)
                accel_x.append(d["values"]["x"])
                accel_y.append(d["values"]["y"])
                accel_z.append(d["values"]["z"])
        return "success", 200
    return "failure", 400

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    app.run_server(debug=True, port=8000, host="0.0.0.0")
