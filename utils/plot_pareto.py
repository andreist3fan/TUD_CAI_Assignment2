import os
from collections import defaultdict

import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_pareto(results_trace: dict, plot_file: str):
    agents = results_trace['connections']
    assert len(agents) == 2

    short_names_dict = {agent : "_".join(agent.split("_")[-2:]) for agent in agents}

    # Initialize data frame for offers with columns (utility_1, utility_2, actor)
    df = pd.DataFrame(columns=[short_names_dict[agents[0]], short_names_dict[agents[1]], "actor"])

    # Initialize dictionary for accepted offers
    accept = {short_names_dict[agents[0]]: [], short_names_dict[agents[1]]: []}

    # Load the data points
    for index, action in enumerate(results_trace["actions"]):
        if "Offer" in action:
            offer = action["Offer"]
            actor = short_names_dict[offer["actor"]]
            new_bid = pd.DataFrame([{
                    short_names_dict[agents[0]]: offer["utilities"][agents[0]],
                    short_names_dict[agents[1]]: offer["utilities"][agents[1]],
                    "actor": actor
                }])
            df = pd.concat([df, new_bid], ignore_index=True)

        elif "Accept" in action:
            offer = action["Accept"]
            for agent, util in offer["utilities"].items():
                accept[short_names_dict[agent]].append(util)

    fig = go.Figure()

    # Plot the offers separately by actor
    color = {0: "red", 1: "blue"}
    for i, agent in enumerate(short_names_dict.values()):
        fig.add_trace(
            go.Scatter(
                x=df[df["actor"] == agent][short_names_dict[agents[0]]],
                y=df[df["actor"] == agent][short_names_dict[agents[1]]],
                name=f"Offered by {agent}",
                mode="markers",
                marker={"color": color[i]},
                hovertemplate='<br>'.join([
                    f"Utility {short_names_dict[agents[0]]}: " + '%{x:.3f}',
                    f"Utility {short_names_dict[agents[1]]}: " + '%{y:.3f}'
                ])
            )
        )

    # Plot the accepted offer
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=accept[short_names_dict[agents[0]]],
            y=accept[short_names_dict[agents[1]]],
            name="Agreement",
            marker={"color": "green", "size": 15},
            hoverinfo="skip"
        )
    )

    # Compute the Pareto frontiers
    sorted_df = df.sort_values(by=short_names_dict[agents[0]], ascending=False)
    pareto_front = []
    max_y = -np.inf
    for _, row in sorted_df.iterrows():
        if row[short_names_dict[agents[1]]] > max_y:
            pareto_front.append(row)
            max_y = row[short_names_dict[agents[1]]]

    # Convert to data frame
    pareto_df = pd.DataFrame(pareto_front)

    # Plot the Pareto frontier line
    fig.add_trace(
        go.Scatter(
            x=pareto_df[short_names_dict[agents[0]]],
            y=pareto_df[short_names_dict[agents[1]]],
            mode="lines",
            name="Pareto Frontier",
            line={"color": "gray", "dash": "dash"},
            hoverinfo="skip"
        )
    )

    # Fix the plot size and the position of the legend
    fig.update_layout(
        width=1000,
        height=900,
        legend={
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
        },
    )

    # Fix the x and y axes and write the figure to html
    fig.update_xaxes(title_text=f"Utility {short_names_dict[agents[0]]}", range=[-0.02, 1.02], ticks="outside")
    fig.update_yaxes(title_text=f"Utility {short_names_dict[agents[1]]}", range=[-0.02, 1.02], ticks="outside")
    fig.write_html(f"{os.path.splitext(plot_file)[0]}.html")
