import json
import time
from pathlib import Path

from utils.plot_trace import plot_trace
from utils.plot_pareto import plot_pareto
from utils.runners import run_session

RESULTS_DIR = Path("results", time.strftime('%Y%m%d-%H%M%S'))

# create results directory if it does not exist
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True)

# Settings to run a negotiation session:
#   You need to specify the classpath of 2 agents to start a negotiation. Parameters for the agent can be added as a dict (see example)
#   You need to specify the preference profiles for both agents. The first profile will be assigned to the first agent.
#   You need to specify a time deadline (is milliseconds (ms)) we are allowed to negotiate before we end without agreement
settings = {
    "agents": [
        {
             "class": "agents.group69_agent.group69_agent.Agent69",
            "parameters": {"storage_dir": "agent_storage/Agent69"},
        },
        {
            "class": "agents.ANL2022.dreamteam109_agent.dreamteam109_agent.DreamTeam109Agent",
        },

    ],
    "profiles": ["domains/domain23/profileA.json", "domains/domain23/profileB.json"],
    "deadline_time_ms": 10000,
}

# {
#             "class": "agents.conceder_agent.conceder_agent.ConcederAgent",
#             "parameters": {"storage_dir": "agent_storage/TemplateAgent"},
#         }
# {
#             "class": "agents.boulware_agent.boulware_agent.BoulwareAgent",
#             "parameters": {"storage_dir": "agent_storage/TemplateAgent"},
#         },
# {
#             "class": "agents.hardliner_agent.hardliner_agent.HardlinerAgent",
#             "parameters": {"storage_dir": "agent_storage/TemplateAgent"},
#         }

# run a session and obtain results in dictionaries
session_results_trace, session_results_summary = run_session(settings)

# plot trace to html file
if not session_results_trace["error"]:
    plot_trace(session_results_trace, RESULTS_DIR.joinpath("trace_plot.html"))
    plot_pareto(session_results_trace, RESULTS_DIR.joinpath("pareto_plot.html"))

# write results to file
with open(RESULTS_DIR.joinpath("session_results_trace.json"), "w", encoding="utf-8") as f:
    f.write(json.dumps(session_results_trace, indent=2))
with open(RESULTS_DIR.joinpath("session_results_summary.json"), "w", encoding="utf-8") as f:
    f.write(json.dumps(session_results_summary, indent=2))
