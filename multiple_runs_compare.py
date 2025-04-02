import subprocess
import re
import plotly.graph_objects as go

NUM_RUNS = 30
utilities = []

for i in range(NUM_RUNS):
    print(f"▶️ Running session #{i + 1}")

    # Run the session script (adjust filename if different)
    result = subprocess.run(
        ["python", "run.py"],
        capture_output=True,
        text=True
    )

    # Extract utility from printed output
    match = re.search(r"We accepted at utility: ([0-9.]+)", result.stdout)
    if match:
        utility = float(match.group(1))
        utilities.append(utility)
        print(f"✅ Utility extracted: {utility}")
    else:
        utilities.append(0.0)
        print("⚠️ Could not extract utility, appended 0.0")

# Plot using Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=list(range(1, NUM_RUNS + 1)),
    y=utilities,
    mode='lines+markers',
    name='Accepted Utility',
    line=dict(color='royalblue'),
    marker=dict(size=8)
))

fig.update_layout(
    title="Accepted Utility Over Multiple Runs",
    xaxis_title="Run Number",
    yaxis_title="Utility",
    xaxis=dict(tickmode='linear'),
    yaxis=dict(range=[0, 1]),
    template="plotly_white"
)

# Save to HTML and display
fig.write_html("learning_vs_dreamteam.html", auto_open=True)
