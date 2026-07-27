from src.rag.report.report_generator import ReportGenerator

generator = ReportGenerator()

def report_agent(state):

    filename = generator.generate(

        prediction=state.get("prediction", "Unknown"),

        confidence=state.get("confidence", 0),

        llm_answer=state["answer"],

        model_name="ResNet18",

    )

    state["report_path"] = str(filename)

    return state