import os
from collections import defaultdict

import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_pareto(results_trace: dict, plot_file: str):
    agents = results_trace['connections']
    assert len(agents) == 2

    # Remove the file path from the agent name
    short_names_dict = {agent : "_".join(agent.split("_")[-2:]) for agent in agents}
    short_names_agents = list(short_names_dict.values())

    # Initialize data frame for offers with columns (utility_1, utility_2, actor)
    df = pd.DataFrame(columns=[short_names_agents[0], short_names_agents[1], "actor"])

    # Initialize dictionary for accepted offers
    accept = {short_names_agents[0]: [], short_names_agents[1]: []}

    # Load the data points
    for index, action in enumerate(results_trace["actions"]):
        if "Offer" in action:
            offer = action["Offer"]
            actor = short_names_dict[offer["actor"]]
            new_bid = pd.DataFrame([{
                    short_names_agents[0]: offer["utilities"][agents[0]],
                    short_names_agents[1]: offer["utilities"][agents[1]],
                    "actor": actor
                }])
            df = pd.concat([df, new_bid], ignore_index=True)

        elif "Accept" in action:
            offer = action["Accept"]
            for agent, util in offer["utilities"].items():
                accept[short_names_dict[agent]].append(util)

    fig = go.Figure()

    # Plot the offers with separate colors by actor
    color = {0: "red", 1: "blue"}
    for i, agent in enumerate(short_names_agents):
        fig.add_trace(
            go.Scatter(
                x=df[df["actor"] == agent][short_names_agents[0]],
                y=df[df["actor"] == agent][short_names_agents[1]],
                name=f"Offered by {agent}",
                mode="markers",
                marker={"color": color[i]},
                hovertemplate='<br>'.join([
                    f"Utility {short_names_agents[0]}: " + '%{x:.3f}',
                    f"Utility {short_names_agents[1]}: " + '%{y:.3f}'
                ])
            )
        )

    # Plot the accepted offer as a green dot
    fig.add_trace(
        go.Scatter(
            mode="markers",
            x=accept[short_names_agents[0]],
            y=accept[short_names_agents[1]],
            name="Agreement",
            marker={"color": "green", "size": 16},
            hoverinfo="skip"
        )
    )

    # Compute the Pareto frontiers
    pareto_frontier = get_pareto_frontier(df, short_names_agents)

    # Compute the Nash points as max(u_1 * u_2)
    nash_points = get_nash_point(pareto_frontier, short_names_agents)

    # Compute the Kalai Smorodinsky point as the point whose ratio u_1 / u_2 is closest to max(u_1) / max(u_2)
    kalai_smorodinsky_point = get_kalai_smorodinsky_point(pareto_frontier, short_names_agents)

    # Compute the Egalitarian point as the offer with the highest minimum utilities for both agents
    egalitarian_point = get_egalitarian_point(pareto_frontier, short_names_agents)

    # Plot the Pareto frontier as a line
    fig.add_trace(
        go.Scatter(
            x=pareto_frontier[short_names_agents[0]],
            y=pareto_frontier[short_names_agents[1]],
            mode="lines",
            name="Pareto Frontier",
            line={"color": "gray", "dash": "dash"},
            hoverinfo="skip"
        )
    )

    # Plot the Nash points
    fig.add_trace(
        go.Scatter(
            x=nash_points[short_names_agents[0]],
            y=nash_points[short_names_agents[1]],
            mode="markers",
            name="Nash Point",
            marker={"color": "orange", "size": 10, "symbol": "diamond"},
            hoverinfo="skip"
        )
    )

    # Plot the Kalai Smorodinsky point
    fig.add_trace(
        go.Scatter(
            x=[kalai_smorodinsky_point[short_names_agents[0]]],
            y=[kalai_smorodinsky_point[short_names_agents[1]]],
            mode="markers",
            name="Kalai Smorodinsky Point",
            marker={"color": "fuchsia", "size": 10, "symbol": "diamond"},
            hoverinfo="skip"
        )
    )

    # Plot the Egalitarian point
    fig.add_trace(
        go.Scatter(
            x=[egalitarian_point[short_names_agents[0]]],
            y=[egalitarian_point[short_names_agents[1]]],
            mode="markers",
            name="Egalitarian Point",
            marker={"color": "indigo", "size": 12, "symbol": "diamond-open-dot"},
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
    fig.update_xaxes(title_text=f"Utility {short_names_agents[0]}", range=[0, 1.02], ticks="outside")
    fig.update_yaxes(title_text=f"Utility {short_names_agents[1]}", range=[0, 1.02], ticks="outside")
    fig.write_html(f"{os.path.splitext(plot_file)[0]}.html")


def get_pareto_frontier(df, short_names_agents):
    # Sort the offers by utility_1
    sorted_df = df.sort_values(by=short_names_agents[0], ascending=False)

    pareto_front = []
    max_y = -np.inf

    # Iterate over the sorted offers and only add them to the frontier if
    # they are better than the current maximum utility_2
    for _, row in sorted_df.iterrows():
        if row[short_names_agents[1]] > max_y:
            pareto_front.append(row)
            max_y = row[short_names_agents[1]]

    return pd.DataFrame(pareto_front)


def get_nash_point(pareto_frontier, short_names_agents):
    # There can be 1 or more Nash points
    nash_points = {short_names_agents[0]: [], short_names_agents[1]: []}
    max_product = 0

    # Calculate the maximum of utility_1 * utility_2
    for _, row in pareto_frontier.iterrows():
        if row[short_names_agents[0]] * row[short_names_agents[1]] > max_product:
            max_product = row[short_names_agents[0]] * row[short_names_agents[1]]

    # Append all Nash points
    for _, row in pareto_frontier.iterrows():
        if row[short_names_agents[0]] * row[short_names_agents[1]] == max_product:
            nash_points[short_names_agents[0]].append(row[short_names_agents[0]])
            nash_points[short_names_agents[1]].append(row[short_names_agents[1]])

    return nash_points


def get_egalitarian_point(pareto_frontier, short_names_agents):
    # Get the offer with the highest minimum utility for both agents
    highest_min_utility_idx = (pareto_frontier[[short_names_agents[0], short_names_agents[1]]].min(axis=1)).idxmax()

    return pareto_frontier.loc[highest_min_utility_idx]


def get_kalai_smorodinsky_point(pareto_frontier, short_names_agents):
    ratio_equal_gain = pareto_frontier[short_names_agents[0]].max() / pareto_frontier[short_names_agents[1]].max()

    closest_ratio_idx = (
            pareto_frontier[short_names_agents[0]] / pareto_frontier[short_names_agents[1]]
            - ratio_equal_gain
    ).abs().idxmin()

    return pareto_frontier.loc[closest_ratio_idx]

