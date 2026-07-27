def report_agent(state):

    state["answer"] = f"""

============================

DIGITAL PATHOLOGY REPORT

============================

Question:

{state["question"]}

--------------------------------

Answer:

{state["answer"]}

"""

    return state